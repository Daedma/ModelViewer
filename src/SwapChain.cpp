
#include <algorithm>

#include "AppInfo.hpp"

#include "SwapChain.hpp"
#include "Device.hpp"
#include "Surface.hpp"
#include "Window.hpp"

SwapChain::SwapChain(const Device& device, vk::RenderPass renderPass) :
	m_device(device),
	m_renderPass(renderPass)
{
	createSwapChain();
}

SwapChain::~SwapChain() noexcept
{
	m_device.get().destroyImageView(m_colorImageView);
	m_device.get().destroyImage(m_colorImage);
	m_device.get().freeMemory(m_colorImageMemory);

	for (auto framebuffer : m_swapChainFramebuffers)
	{
		m_device.get().destroyFramebuffer(framebuffer);
	}

	for (auto imageView : m_swapChainImageViews)
	{
		m_device.get().destroyImageView(imageView);
	}

	m_device.get().destroySwapchainKHR(m_swapChain);
}

vk::Format SwapChain::getImageFormat(const Device& device)
{
	Device::SwapChainSupportDetails swapChainSupport = device.querySwapChainSupport();
	vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(swapChainSupport.formats);
	return surfaceFormat.format;
}

void SwapChain::createSwapChain()
{
	Device::SwapChainSupportDetails swapChainSupport = m_device.querySwapChainSupport();

	vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(swapChainSupport.formats);
	vk::PresentModeKHR presentMode = choosePresentMode(swapChainSupport.presentModes);
	vk::Extent2D extent = chooseExtent(swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 &&
		imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR createInfo({},
		m_device.getSurface(),
		imageCount,
		surfaceFormat.format,
		surfaceFormat.colorSpace,
		extent,
		1,
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::SharingMode::eExclusive,
		nullptr,
		swapChainSupport.capabilities.currentTransform,
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		presentMode,
		vk::True);

	Device::QueueFamilyIndices indices = m_device.findQueueFamilies();
	std::array<uint32_t, 2> queueFamilyIndices{ indices.graphicsFamily.value(), indices.presentFamily.value() };

	if (indices.graphicsFamily != indices.presentFamily)
	{
		createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
		createInfo.setQueueFamilyIndices(queueFamilyIndices);
	}

	m_swapChain = m_device.get().createSwapchainKHR(createInfo);
	m_swapChainImages = m_device.get().getSwapchainImagesKHR(m_swapChain);
	m_swapChainImageFormat = surfaceFormat.format;
	m_swapChainExtent = extent;
}

vk::SurfaceFormatKHR SwapChain::chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	vk::SurfaceFormatKHR usedFormat = { appInfo::features::imageFormat, appInfo::features::colorSpace };
	if (std::find(availableFormats.cbegin(), availableFormats.cend(), usedFormat) == availableFormats.cend())
	{
		throw std::runtime_error("No suitable surface format found.");
	}
	return usedFormat;
}

vk::PresentModeKHR SwapChain::choosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const
{
	for (const auto& availablePresentMode : availablePresentModes)
	{
		if (availablePresentMode == vk::PresentModeKHR::eMailbox)
		{
			return availablePresentMode;
		}
	}
	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D SwapChain::chooseExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return capabilities.currentExtent;
	}
	else
	{
		vk::Extent2D actualExtent = m_device.getSurface().getWindow().getFramebufferExtent();
		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
		return actualExtent;
	}
}