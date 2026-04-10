output "vpc_id" {
  value = aws_vpc.main.id
}

output "alb_dns_name" {
  value = aws_lb.main.dns_name
}

output "data_lake_bucket" {
  value = aws_s3_bucket.data_lake.bucket
}

output "audit_table_name" {
  value = aws_dynamodb_table.audit_log.name
}

output "track_a_queue_url" {
  value = aws_sqs_queue.track_a.url
}

output "track_b_queue_url" {
  value = aws_sqs_queue.track_b.url
}

output "review_queue_url" {
  value = aws_sqs_queue.review.url
}
