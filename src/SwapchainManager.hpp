#pragma once
#include <vector>
#include <vulkan/vulkan_raii.hpp>

class VulkanContext;

class SwapchainManager
{
public:
  SwapchainManager( VulkanContext & ctx, VkSurfaceKHR surface, uint32_t width, uint32_t height );
  void recreate( uint32_t newWidth, uint32_t newHeight );

  const vk::raii::SwapchainKHR &       getSwapchain() const noexcept;
  const std::vector<vk::ImageView> &   getImageViews() const noexcept;
  const std::vector<vk::Framebuffer> & getFramebuffers() const noexcept;

private:
  VulkanContext & context;

  vk::raii::SwapchainKHR       swapchain;
  std::vector<vk::Image>       images;
  std::vector<vk::ImageView>   imageViews;
  std::vector<vk::Framebuffer> framebuffers;

  void createSwapchain( uint32_t width, uint32_t height );
  void createImageViews();
  void createFramebuffers();
};
