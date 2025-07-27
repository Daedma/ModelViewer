#pragma once

#include <vulkan/vulkan.hpp>

#include <optional>
#include <vector>

class Surface;

class Device
{
public:
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
	Device(vk::Instance instance, const Surface& surface);

	~Device() noexcept;

	vk::PhysicalDevice getPhysicalDevice() const noexcept { return m_physicalDevice; }

	vk::Device get() const noexcept { return m_logicalDevice; }

	const Surface& getSurface() const noexcept { return m_surface; }

	vk::Queue getGraphicsQueue() const noexcept { return m_graphicsQueue; }

	vk::Queue getPresentQueue() const noexcept { return m_presentQueue; }

	vk::SampleCountFlagBits getMsaaSamples() const noexcept { return m_msaaSamples; }

	SwapChainSupportDetails querySwapChainSupport() const
	{
		return querySwapChainSupport(m_physicalDevice);
	}

	QueueFamilyIndices findQueueFamilies() const
	{
		return findQueueFamilies(m_physicalDevice);
	}

private:
	vk::Instance m_instance;
	const Surface& m_surface;

	// TODO vk::QueueFamilyIndices m_queueFamilies;
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