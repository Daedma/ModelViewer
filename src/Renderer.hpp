#pragma once
#include "SwapchainManager.hpp"
#include "VulkanContext.hpp"

#include <vector>
#include <vulkan/vulkan_raii.hpp>

class Renderer
{
public:
    Renderer(VulkanContext& ctx, SwapchainManager& swapchain);

    void drawFrame();

private:
    VulkanContext&    context;
    SwapchainManager& swapchain;

    vk::raii::CommandPool                commandPool;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::Semaphore>     imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore>     renderFinishedSemaphores;
    std::vector<vk::raii::Fence>         inFlightFences;

    void createCommandBuffers();
    void recordCommandBuffer(uint32_t imageIndex);
};
