variable "project_name" {
  description = "Project prefix for resource naming."
  type        = string
  default     = "nlp-uk"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region for deployment."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR for VPC."
  type        = string
  default     = "10.40.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDRs for public subnets."
  type        = list(string)
  default     = ["10.40.1.0/24", "10.40.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDRs for private subnets."
  type        = list(string)
  default     = ["10.40.11.0/24", "10.40.12.0/24"]
}

variable "container_image_review_ui" {
  description = "Container image for review UI service."
  type        = string
}

variable "container_image_worker" {
  description = "Container image for worker service."
  type        = string
}

variable "notification_email" {
  description = "Email for critical alerts."
  type        = string
}
