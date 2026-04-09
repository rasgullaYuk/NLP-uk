"""
bedrock_client.py — Tier 3 OCR Correction Module
=================================================
Handles all communication with AWS Bedrock (Claude Sonnet, vision-capable).

Public API
----------
    bedrock_call(ocr_text, image, bbox, surrounding_context, region_index)
        → dict: { corrected_text, confidence, reasoning }  OR  raises

Design decisions
----------------
* Uses the *Messages* API (anthropic.claude-3-x) with an image block + text block.
* Image is cropped from the full page PIL Image using the region's bbox.
  Fallback: if crop produces an empty/invalid image, the full page is used.
* Surrounding context is truncated to CONTEXT_MAX_TOKENS before embedding
  in the prompt to avoid runaway token usage.
* Exponential backoff retries on throttling / transient errors.
* Hard timeout enforced via `socket.setdefaulttimeout` before each call,
  restored afterwards.  If the call exceeds BEDROCK_CALL_TIMEOUT_SECONDS,
  a TimeoutError is raised (caller maps this to TIMEOUT reason code).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import socket
import time
from typing import Any

from botocore.exceptions import ClientError
from PIL import Image
try:
    from hipaa_compliance import create_secure_client
except ImportError:
    import boto3

    def create_secure_client(service_name: str, region_name: str, **kwargs):
        return boto3.client(service_name, region_name=region_name, **kwargs)

try:
    from .config import (
        AWS_REGION,
        BEDROCK_MODEL_ID,
        BEDROCK_CALL_TIMEOUT_SECONDS,
        CONTEXT_MAX_TOKENS,
        MAX_RESPONSE_TOKENS,
        MAX_RETRIES,
        RETRY_BASE_DELAY_SECONDS,
        WORDS_PER_TOKEN_ESTIMATE,
    )
except ImportError:
    # Fallback for direct execution
    from config import (
        AWS_REGION,
        BEDROCK_MODEL_ID,
        BEDROCK_CALL_TIMEOUT_SECONDS,
        CONTEXT_MAX_TOKENS,
        MAX_RESPONSE_TOKENS,
        MAX_RETRIES,
        RETRY_BASE_DELAY_SECONDS,
        WORDS_PER_TOKEN_ESTIMATE,
    )

logger = logging.getLogger(__name__)

# ── Bedrock client (module-level singleton — thread-safe for read) ─────────────
_bedrock_runtime: Any = None


def _get_client() -> Any:
    """Lazy-initialise the Bedrock Runtime boto3 client."""
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = create_secure_client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_runtime


# ── Context window truncation ─────────────────────────────────────────────────

def _truncate_context(context_text: str, max_tokens: int = CONTEXT_MAX_TOKENS) -> str:
    """
    Restrict surrounding_context_text to approximately max_tokens.

    Uses a word-count sliding window centred on the text.  The estimate is
    conservative (WORDS_PER_TOKEN_ESTIMATE words per token).

    Args:
        context_text: Raw surrounding context string.
        max_tokens:   Upper token budget (default from config).

    Returns:
        Possibly-truncated context string.
    """
    if not context_text:
        return ""

    max_words = int(max_tokens / WORDS_PER_TOKEN_ESTIMATE)
    words = context_text.split()

    if len(words) <= max_words:
        return context_text

    # Take words from the centre of the provided context for maximum relevance.
    mid = len(words) // 2
    half = max_words // 2
    start = max(0, mid - half)
    end = min(len(words), start + max_words)
    truncated = " ".join(words[start:end])

    logger.debug(
        "Context truncated from %d words to %d words (token budget: %d).",
        len(words), len(truncated.split()), max_tokens,
    )
    return truncated


# ── Image preparation ─────────────────────────────────────────────────────────

def _crop_image(full_page_image: Image.Image, bbox: list) -> Image.Image:
    """
    Crop the full-page PIL Image to the region's bounding box.

    Falls back to the full page image if:
    - bbox is None / empty
    - The resulting crop would be degenerate (zero width or height)
    - Any exception occurs during cropping

    Args:
        full_page_image: PIL Image of the full document page.
        bbox:            [x1, y1, x2, y2] in pixel coordinates.

    Returns:
        A PIL Image (either the crop or the full page as fallback).
    """
    try:
        if not bbox or len(bbox) < 4:
            raise ValueError("Invalid bbox — using full page.")

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Clamp to image dimensions
        img_w, img_h = full_page_image.size
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate bbox after clamping: {[x1, y1, x2, y2]}")

        cropped = full_page_image.crop((x1, y1, x2, y2))
        logger.debug("Cropped region bbox=%s size=%s", bbox, cropped.size)
        return cropped

    except Exception as exc:
        logger.warning(
            "Bbox crop failed (%s). Falling back to full page image.", exc
        )
        return full_page_image


def _image_to_base64(image: Image.Image) -> tuple[str, str]:
    """
    Encode a PIL Image as a base64 PNG string for the Bedrock Messages API.

    Args:
        image: PIL Image object.

    Returns:
        Tuple of (base64_string, media_type).  media_type is always "image/png".
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.standard_b64encode(buffer.read()).decode("utf-8")
    return b64, "image/png"


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_prompt(ocr_text: str, context_text: str) -> str:
    """
    Construct the clinical correction prompt.

    The prompt explicitly:
    - Instructs the model to correct ONLY if the OCR text is clearly wrong.
    - Prohibits any medical hallucination or invention.
    - Requires a strict JSON response (no markdown, no prose outside JSON).

    Args:
        ocr_text:     The low-confidence OCR string to evaluate.
        context_text: Truncated surrounding clinical text for reference.

    Returns:
        The formatted prompt string.
    """
    return f"""You are a clinical document OCR correction assistant.

## Task
You have been given a snippet of text extracted by OCR that has LOW confidence. Your only job is to correct clear OCR errors (misidentified characters, broken words) using the surrounding clinical context as a guide.

## Strict Rules
1. ONLY correct the text if there is an obvious OCR error (e.g., "rn" misread as "m", "0" vs "O", broken ligatures).
2. DO NOT invent, add, or fabricate any medical terms, drug names, dosages, or clinical findings.
3. DO NOT change abbreviations unless they are unambiguously wrong OCR artifacts.
4. If the text looks correct as-is, return it unchanged.
5. Your response MUST be valid JSON only — no markdown fences, no explanation outside the JSON object.

## Surrounding Clinical Context (for reference only — do NOT copy from it)
{context_text if context_text else "No surrounding context available."}

## Low-Confidence OCR Text to Evaluate
{ocr_text}

## Required JSON Response Format
{{
  "corrected_text": "<corrected or unchanged text>",
  "confidence": <float between 0.0 and 1.0 indicating your confidence in the correction>,
  "reasoning": "<one sentence explaining what was corrected and why, or why no change was made>"
}}"""


# ── Core Bedrock call ─────────────────────────────────────────────────────────

def bedrock_call(
    ocr_text:            str,
    full_page_image:     Image.Image,
    bbox:                list,
    surrounding_context: str,
    region_index:        int = 0,
) -> dict[str, Any]:
    """
    Call AWS Bedrock (Claude Sonnet vision) to propose a correction for a
    single low-confidence OCR region.

    Args:
        ocr_text:            The low-confidence text string from Textract.
        full_page_image:     Full-page PIL Image.  Will be cropped internally.
        bbox:                [x1, y1, x2, y2] bounding box of the region.
        surrounding_context: Raw surrounding context text (will be truncated).
        region_index:        Zero-based index of the region (for logging).

    Returns:
        dict with keys:
            - corrected_text  (str)
            - confidence      (float, 0–1)
            - reasoning       (str)

    Raises:
        TimeoutError:    If the Bedrock call exceeds BEDROCK_CALL_TIMEOUT_SECONDS.
        RuntimeError:    If all retries are exhausted without a valid response.
        ValueError:      If the model response cannot be parsed as valid JSON.
    """
    client = _get_client()

    # Prepare image payload
    cropped_image = _crop_image(full_page_image, bbox)
    b64_image, media_type = _image_to_base64(cropped_image)

    # Truncate context
    truncated_context = _truncate_context(surrounding_context)

    # Build prompt
    prompt_text = _build_prompt(ocr_text, truncated_context)

    # Assemble Bedrock Messages API payload
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_RESPONSE_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": media_type,
                            "data":       b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
    }

    last_exception: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(
            "Bedrock call — region %d, attempt %d/%d.", region_index, attempt, MAX_RETRIES
        )

        # Apply socket-level timeout for the HTTP call
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(BEDROCK_CALL_TIMEOUT_SECONDS)

        try:
            response = client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            raw_body = response["body"].read().decode("utf-8")
            response_data = json.loads(raw_body)

            # Extract the text block from the response
            content_blocks = response_data.get("content", [])
            model_text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    model_text = block["text"].strip()
                    break

            if not model_text:
                raise ValueError("Model returned empty content block.")

            # Parse the model's JSON response
            parsed: dict = json.loads(model_text)

            corrected_text = str(parsed.get("corrected_text", ocr_text)).strip()
            llm_confidence = float(parsed.get("confidence", 0.0))
            reasoning      = str(parsed.get("reasoning", "")).strip()

            # Clamp confidence to [0, 1]
            llm_confidence = max(0.0, min(1.0, llm_confidence))

            logger.info(
                "Region %d: correction='%s', llm_confidence=%.2f",
                region_index, corrected_text[:60], llm_confidence,
            )

            return {
                "corrected_text": corrected_text,
                "confidence":     llm_confidence,
                "reasoning":      reasoning,
            }

        except socket.timeout as exc:
            logger.warning("Bedrock call timed out on attempt %d: %s", attempt, exc)
            last_exception = TimeoutError(
                f"Bedrock call exceeded {BEDROCK_CALL_TIMEOUT_SECONDS}s timeout."
            )
            break  # Timeout is non-retryable — exit immediately

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("ThrottlingException", "ServiceUnavailableException"):
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Throttled on attempt %d. Retrying in %.1fs.", attempt, delay
                )
                last_exception = exc
                time.sleep(delay)
            else:
                logger.error("Non-retryable Bedrock ClientError: %s", exc)
                raise

        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse model JSON on attempt %d: %s. Raw: %.200s",
                attempt, exc, model_text if "model_text" in dir() else "(no response)",
            )
            last_exception = exc
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay)

        except Exception as exc:
            logger.error("Unexpected error on attempt %d: %s", attempt, exc)
            last_exception = exc
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay)

        finally:
            # Always restore original socket timeout
            socket.setdefaulttimeout(original_timeout)

    # All retries exhausted
    if isinstance(last_exception, TimeoutError):
        raise last_exception

    raise RuntimeError(
        f"All {MAX_RETRIES} Bedrock attempts failed for region {region_index}. "
        f"Last error: {last_exception}"
    )
