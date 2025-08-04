#include "VulkanContextBuilder.hpp"

#include "Utils.hpp"

#include <GLFW/glfw3.h>
#include <cstring>

VulkanContext VulkanContextBuilder::build()
{
  vk::raii::Context  context;
  vk::raii::Instance instance = createInstance( context );
}

vk::raii::Instance VulkanContextBuilder::createInstance( vk::raii::Context & context )
{
  if ( m_enableValidation && !checkValidationLayersSupport() )
  {
    throw std::runtime_error( "validation layers requested, but not available!" );
  }

  vk::ApplicationInfo       appInfo( m_appName.c_str(), m_appVersion, m_engineName.c_str(), m_engineVersion, m_apiVersion );
  std::vector<const char *> extensions = getRequiredExtensions();
  vk::InstanceCreateInfo    instanceCreateInfo( {}, &appInfo, {}, extensions );

  vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo( {}, m_messageSeverity, m_messageType, debugCallback );
  if ( m_enableValidation )
  {
    instanceCreateInfo.setPEnabledExtensionNames( m_validationLayers );
    instanceCreateInfo.setPNext( &debugMessengerCreateInfo );
  }

  return context.createInstance( instanceCreateInfo );
}

std::vector<const char *> VulkanContextBuilder::getRequiredExtensions() const
{
  uint32_t      glfwExtensionCount = 0;
  const char ** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions( &glfwExtensionCount );

  std::vector<const char *> extensions( glfwExtensions, glfwExtensions + glfwExtensionCount );

  if ( enableValidationLayers )
  {
    extensions.emplace_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );
  }

  return extensions;
}

bool VulkanContextBuilder::checkValidationLayersSupport()
{
  auto availableLayers = vk::enumerateInstanceLayerProperties();

  for ( const char * layerName : m_validationLayers )
  {
    bool layerFound = false;

    for ( const auto & layerProperties : availableLayers )
    {
      if ( strcmp( layerName, layerProperties.layerName ) == 0 )
      {
        layerFound = true;
        break;
      }
    }

    if ( !layerFound )
    {
      return false;
    }
  }

  return true;
}
