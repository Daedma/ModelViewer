#include "VulkanContext.hpp"

VulkanContext::VulkanContext() :
  m_context(),
  
{
  createInstance();
  selectPhysicalDevice();
  createLogicalDevice();
}
