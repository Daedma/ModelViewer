#pragma once

#include <vulkan/vulkan.hpp>

#include <optional>
#include <vector>

class Device
{
	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete()
		{
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	struct SwapChainSupportDetails
	{
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

public:
	Device(vk::Instance instance, vk::SurfaceKHR surface);

	~Device() noexcept;
private:
	vk::Instance m_instance;
	vk::SurfaceKHR m_surface;

	vk::PhysicalDevice m_physicalDevice;
	vk::SampleCountFlagBits m_msaaSamples;

	vk::Device m_logicalDevice;

	vk::Queue m_graphicsQueue;
	vk::Queue m_presentQueue;

private:
	static const std::vector<const char*> deviceExtensions;

private:
	vk::PhysicalDevice pickPhysicalDevice() const;

	bool isDeviceSuitable(vk::PhysicalDevice device) const;

	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) const;

	bool checkDeviceExtensionSupport(vk::PhysicalDevice device) const;

	SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device) const;

	vk::SampleCountFlagBits getMaxUsableSampleCount() const;

	vk::Device createLogicalDevice() const;
};