#pragma once

#include <vulkan/vulkan.hpp>

#include <vector>

class Device;

class SwapChain
{
public:
	SwapChain(const Device& device, vk::RenderPass renderPass);

	~SwapChain() noexcept;

	vk::SwapchainKHR get() const noexcept { return m_swapChain; }

private:
	const Device& m_device;

	vk::RenderPass m_renderPass;

	vk::SwapchainKHR m_swapChain;
	std::vector<vk::Image> m_swapChainImages;
	vk::Format m_swapChainImageFormat;
	vk::Extent2D m_swapChainExtent;

	std::vector<vk::ImageView> m_swapChainImageViews;
	std::vector<vk::Framebuffer> m_swapChainFramebuffers;

private:
	void createSwapChain();

	vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);

	vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const;

	vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const;

};