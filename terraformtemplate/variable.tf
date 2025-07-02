variable "name" {}
variable "resource_group_name" {}
variable "environment_id" {}
variable "image" {}
variable "cpu" {}
variable "memory" {}

variable "env_azure_storage_connection_string" {}
variable "env_azure_container_name" {}
variable "env_azure_storage_account" {}
variable "env_provider" {}

variable "secret_name" {}
variable "secret_value" {}

variable "registry_server" {}
variable "registry_username" {}

variable "ingress_external" {
  type = bool
}
