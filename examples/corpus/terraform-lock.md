Terraform state locking prevents two applies running at once. DynamoDB provides the lock
for the S3 backend. A stale lock left behind by a crashed run can be cleared with
force-unlock, but only once you are certain no apply is still in flight.
