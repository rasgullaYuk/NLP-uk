resource "aws_sqs_queue" "track_a_dlq" {
  name                      = "${local.name_prefix}-track-a-dlq"
  message_retention_seconds = 1209600
  tags                      = local.common_tags
}

resource "aws_sqs_queue" "track_a" {
  name                       = "${local.name_prefix}-track-a"
  visibility_timeout_seconds = 300
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.track_a_dlq.arn
    maxReceiveCount     = 3
  })
  tags = local.common_tags
}

resource "aws_sqs_queue" "track_b" {
  name                       = "${local.name_prefix}-track-b"
  visibility_timeout_seconds = 300
  tags                       = local.common_tags
}

resource "aws_sqs_queue" "review" {
  name = "${local.name_prefix}-review"
  tags = local.common_tags
}

resource "aws_sns_topic" "critical_alerts" {
  name = "${local.name_prefix}-critical-alerts"
  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.critical_alerts.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/${local.name_prefix}/app"
  retention_in_days = 30
  tags              = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "queue_depth_alarm" {
  alarm_name          = "${local.name_prefix}-track-a-queue-depth"
  alarm_description   = "Track A queue depth > 100"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.track_a.name }
  statistic           = "Average"
  period              = 60
  evaluation_periods  = 1
  threshold           = 100
  comparison_operator = "GreaterThanThreshold"
  alarm_actions       = [aws_sns_topic.critical_alerts.arn]
  tags                = local.common_tags
}
