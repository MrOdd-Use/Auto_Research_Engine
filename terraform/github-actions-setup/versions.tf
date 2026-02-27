terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket         = "gg-ai-terraform-states"
    key            = "iam-setup/github-actions-Auto_Research_Engine/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = "us-east-1"
}
