terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_container_app" "inference_model" {
  name                          = var.name
  resource_group_name           = var.resource_group_name
  container_app_environment_id  = var.environment_id
  workload_profile_name         = "Consumption"
  revision_mode                 = "Single"

  template {
    container {
      name   = var.name
      image  = var.image
      cpu    = var.cpu
      memory = var.memory

      env {
        name  = "AZURE_STORAGE_CONNECTION_STRING"
        value = var.env_azure_storage_connection_string
      }

      env {
        name  = "AZURE_CONTAINER_NAME"
        value = var.env_azure_container_name
      }

      env {
        name  = "AZURE_STORAGE_ACCOUNT"
        value = var.env_azure_storage_account
      }

      env {
        name  = "PROVIDER"
        value = var.env_provider
      }
    }

    min_replicas = 0
  }

  identity {
    type = "SystemAssigned"
  }

  secret {
    name  = var.secret_name
    value = var.secret_value
  }

  registry {
    server               = var.registry_server
    username             = var.registry_username
    password_secret_name = var.secret_name
  }

  ingress {
    external_enabled = var.ingress_external
    target_port      = 8000

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}
