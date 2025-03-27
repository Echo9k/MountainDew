# main.tf
provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "llm" {
  name     = "llm-rg"
  location = "West US"
}

module "aks_cluster" {
  source = "Azure/aks/azurerm"
  name                = "llm-aks"
  resource_group_name = azurerm_resource_group.llm.name
  kubernetes_version  = "1.29"
  node_pools = [
    {
      name         = "cpu-pool"
      vm_size      = "Standard_DS2_v2"
      node_count   = 1
      max_pods     = 30
    }
  ]
  azure_policy_enabled = true
}
