#include "VulkanContextBuilder.hpp"

#include "Utils.hpp"

#include <GLFW/glfw3.h>
#include <cstring>

VulkanContext VulkanContextBuilder::build()
{
  vk::raii::Context  context;
  vk::raii::Instance instance = createInstance( context );

  vk::raii::DebugUtilsMessengerEXT debugMessenger( nullptr );
  if ( m_enableValidation )
  {
    vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo( {}, m_messageSeverity, m_messageType, debugCallback );
    debugMessenger = instance.createDebugUtilsMessengerEXT( debugMessengerCreateInfo );
  }

  std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
  if ( physicalDevices.empty() )
  {
    throw std::runtime_error{ "There is no Vulkan-compatible devices!" };
  }
  // TODO : make a check for compabilities like in tutorial
  vk::raii::PhysicalDevice physicalDevice = physicalDevices.front();

  uint32_t                  graphicsQueueIndex = chooseGraphicsQueue( physicalDevice );
  vk::DeviceQueueCreateInfo queueCreateInfo( {}, graphicsQueueIndex, 1.f );
  vk::DeviceCreateInfo      deviceCreateInfo( {}, queueCreateInfo, {}, m_deviceExtensions );
  vk::raii::Device          device        = physicalDevice.createDevice( deviceCreateInfo );
  vk::raii::Queue           graphicsQueue = device.getQueue( graphicsQueueIndex, 0 );

  return VulkanContext{ std::move( context ), std::move( instance ), std::move( physicalDevice ), std::move( device ), std::move( graphicsQueue ) };
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
    instanceCreateInfo.setPEnabledLayerNames( m_validationLayers );
    instanceCreateInfo.setPNext( &debugMessengerCreateInfo );
  }

  return context.createInstance( instanceCreateInfo );
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

std::vector<const char *> VulkanContextBuilder::getRequiredExtensions() const
{
  uint32_t      glfwExtensionCount = 0;
  const char ** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions( &glfwExtensionCount );

  std::vector<const char *> extensions( glfwExtensions, glfwExtensions + glfwExtensionCount );

  if ( m_enableValidation )
  {
    extensions.emplace_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );
  }

  return extensions;
}

uint32_t VulkanContextBuilder::chooseGraphicsQueue( const vk::raii::PhysicalDevice & physicalDevice ) const
{
  std::vector<vk::QueueFamilyProperties> qfProperties = physicalDevice.getQueueFamilyProperties();

  for ( uint32_t i = 0; i != qfProperties.size(); ++i )
  {
    if ( qfProperties[i].queueFlags & vk::QueueFlagBits::eGraphics )
    {
      return i;
    }
  }
  auto deviceProperties = physicalDevice.getProperties();
  throw std::runtime_error{ "There is no any graphics queue for device " + std::string{ deviceProperties.deviceName.data() } +
                            "(id : " + std::to_string( deviceProperties.deviceID ) + ")" };
}
