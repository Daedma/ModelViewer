// VulkanContextBuilder.hpp
#pragma once
#include "VulkanContext.hpp"

#include <string>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

class VulkanContextBuilder
{
public:
  VulkanContextBuilder & setAppName( const std::string & name )
  {
    m_appName = name;
    return *this;
  }

  VulkanContextBuilder & setAppVersion( uint32_t major, uint32_t minor, uint32_t patch )
  {
    m_appVersion = VK_MAKE_API_VERSION( 0, major, minor, patch );
    return *this;
  }

  VulkanContextBuilder & setEngineName( const std::string & name )
  {
    m_engineName = name;
    return *this;
  }

  VulkanContextBuilder & setEngineVersion( uint32_t major, uint32_t minor, uint32_t patch )
  {
    m_engineVersion = VK_MAKE_API_VERSION( 0, major, minor, patch );
    return *this;
  }

  VulkanContextBuilder & setApiVersion( uint32_t major, uint32_t minor, uint32_t patch )
  {
    m_apiVersion = VK_MAKE_API_VERSION( 0, major, minor, patch );
    return *this;
  }

  VulkanContextBuilder & enableValidationLayers( bool enable )
  {
    m_enableValidation = enable;
    return *this;
  }

  VulkanContextBuilder & setValidationMessageSeverity( vk::DebugUtilsMessageSeverityFlagsEXT severity )
  {
    m_messageSeverity = severity;
    return *this;
  }

  VulkanContextBuilder & setValidationMessageType( vk::DebugUtilsMessageTypeFlagsEXT type )
  {
    m_messageType = type;
    return *this;
  }

  VulkanContextBuilder & addDeviceExtension( const char * extension )
  {
    m_deviceExtensions.emplace_back( extension );
    return *this;
  }

  VulkanContext build();

private:
  std::string                           m_appName          = "VulkanApp";
  uint32_t                              m_appVersion       = VK_MAKE_API_VERSION( 0, 1, 0, 0 );
  std::string                           m_engineName       = "No Engine";
  uint32_t                              m_engineVersion    = VK_MAKE_API_VERSION( 0, 1, 0, 0 );
  uint32_t                              m_apiVersion       = VK_API_VERSION_1_3;
  bool                                  m_enableValidation = true;
  std::vector<const char *>             m_validationLayers = { "VK_LAYER_KHRONOS_validation" };
  std::vector<const char *>             m_deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
  vk::DebugUtilsMessageSeverityFlagsEXT m_messageSeverity =
    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
  vk::DebugUtilsMessageTypeFlagsEXT m_messageType =
    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

  vk::raii::Instance        createInstance( vk::raii::Context & context );
  std::vector<const char *> getRequiredExtensions() const;
  bool                      checkValidationLayersSupport();

  uint32_t chooseGraphicsQueue(const vk::raii::PhysicalDevice& physicalDevice) const;
};
