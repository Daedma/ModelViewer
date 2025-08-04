// VulkanContext.hpp
#pragma once
#include <memory>
#include <vulkan/vulkan_raii.hpp>

class VulkanContext
{
public:
  const vk::raii::Instance & getInstance() const noexcept
  {
    return m_instance;
  }

  const vk::raii::PhysicalDevice & getPhysicalDevice() const noexcept
  {
    return m_physicalDevice;
  }

  const vk::raii::Device & getDevice() const noexcept
  {
    return m_device;
  }

  const vk::raii::Queue & getGraphicsQueue() const noexcept
  {
    return m_graphicsQueue;
  }

private:
  vk::raii::Context        m_context;
  vk::raii::Instance       m_instance;
  vk::raii::PhysicalDevice m_physicalDevice;
  vk::raii::Device         m_device;
  vk::raii::Queue          m_graphicsQueue;

  VulkanContext( vk::raii::Context &&        ctx,
                 vk::raii::Instance &&       instance,
                 vk::raii::PhysicalDevice && physicalDevice,
                 vk::raii::Device &&         device,
                 vk::raii::Queue &&          graphicsQueue )
    : m_context( std::move( ctx ) )
    , m_instance( std::move( instance ) )
    , m_physicalDevice( std::move( physicalDevice ) )
    , m_device( std::move( device ) )
    , m_graphicsQueue( std::move( graphicsQueue ) )
  {
  }

  friend class VulkanContextBuilder;
};
