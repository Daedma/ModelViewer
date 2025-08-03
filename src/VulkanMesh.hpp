#pragma once
#include <vulkan/vulkan_raii.hpp>

class VulkanMesh
{
public:
  VulkanMesh( vk::raii::Device & device, vk::raii::PhysicalDevice & physicalDevice );
  void uploadVertexData( const std::vector<float> & vertices );

private:
  vk::raii::Buffer       vertexBuffer;
  vk::raii::DeviceMemory memory;
};
