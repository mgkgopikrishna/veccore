IRSA lets a pod assume an AWS IAM role through a projected service account token, so no
long-lived AWS credential is ever stored in the cluster. It is the preferred alternative
to a mounted access key.
