#pragma once

#include <vulkan/vulkan_raii.hpp>

class VulkanContext
{
public:
  VulkanContext();
  ~VulkanContext() noexcept;

  const vk::raii::Instance &       getInstance() const noexcept;
  const vk::raii::Device &         getDevice() const noexcept;
  const vk::raii::PhysicalDevice & getPhysicalDevice() const noexcept;
  const vk::raii::Queue &          getGraphicsQueue() const noexcept;

private:
  vk::raii::Context        m_context;
  vk::raii::Instance       m_instance;
  vk::raii::Device         m_device;
  vk::raii::PhysicalDevice m_physicalDevice;
  vk::raii::Queue          m_graphicsQueue;

  void createInstance();
  void selectPhysicalDevice();
  void createLogicalDevice();
};
