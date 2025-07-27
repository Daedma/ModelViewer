#include <set>
#include <string>

#include "Device.hpp"
#include "Surface.hpp"

#include "AppInfo.hpp"

const std::vector<const char*> Device::deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

Device::Device(vk::Instance instance, const Surface& surface) :
	m_instance(instance),
	m_surface(surface),
	m_physicalDevice(pickPhysicalDevice()),
	m_msaaSamples(getMaxUsableSampleCount()),
	m_logicalDevice(createLogicalDevice())
{
	QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
	m_graphicsQueue = m_logicalDevice.getQueue(indices.graphicsFamily.value(), 0);
	m_presentQueue = m_logicalDevice.getQueue(indices.presentFamily.value(), 0);
}

Device::~Device() noexcept
{
	m_logicalDevice.destroy();
}

vk::PhysicalDevice Device::pickPhysicalDevice() const
{
	auto devices = m_instance.enumeratePhysicalDevices();
	if (devices.empty())
	{
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}
	for (const auto& device : devices)
	{
		if (isDeviceSuitable(device))
		{
			return device;
		}
	}
	throw std::runtime_error("failed to find a suitable GPU!");
}

bool Device::isDeviceSuitable(vk::PhysicalDevice device) const
{
	QueueFamilyIndices indices = findQueueFamilies(device);

	bool extensionsSupported = checkDeviceExtensionSupport(device);

	bool swapChainAdequate = false;
	if (extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupport.formats.empty() &&
			!swapChainSupport.presentModes.empty();
	}

	vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

	return indices.isComplete() &&
		extensionsSupported &&
		swapChainAdequate &&
		supportedFeatures.samplerAnisotropy;
}

Device::QueueFamilyIndices Device::findQueueFamilies(vk::PhysicalDevice device) const
{
	QueueFamilyIndices indices;

	auto queueFamilies = device.getQueueFamilyProperties();

	int i = 0;
	for (const auto& queueFamily : queueFamilies)
	{
		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
		{
			indices.graphicsFamily = i;
		}

		vk::Bool32 presentSupport = device.getSurfaceSupportKHR(i, m_surface.get());

		if (presentSupport)
		{
			indices.presentFamily = i;
		}

		if (indices.isComplete())
		{
			break;
		}

		++i;
	}

	return indices;
}

bool Device::checkDeviceExtensionSupport(vk::PhysicalDevice device) const
{
	auto availableExtensions = device.enumerateDeviceExtensionProperties();
	std::set<std::string> requiredExtensions(std::begin(deviceExtensions), std::end(deviceExtensions));

	for (const auto& extension : availableExtensions)
	{
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

Device::SwapChainSupportDetails Device::querySwapChainSupport(vk::PhysicalDevice device) const
{
	SwapChainSupportDetails details;
	details.capabilities = device.getSurfaceCapabilitiesKHR(m_surface.get());
	details.formats = device.getSurfaceFormatsKHR(m_surface.get());
	details.presentModes = device.getSurfacePresentModesKHR(m_surface.get());
	return details;
}

vk::SampleCountFlagBits Device::getMaxUsableSampleCount() const
{
	auto physicalDeviceProperties = m_physicalDevice.getProperties();

	vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts &
		physicalDeviceProperties.limits.framebufferDepthSampleCounts;
	if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
	if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
	if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
	if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
	if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
	if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

	return vk::SampleCountFlagBits::e1;
}

vk::Device Device::createLogicalDevice() const
{
	QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

	const float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies)
	{
		queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlagBits{}, queueFamily, 1, &queuePriority, nullptr);
	}

	vk::PhysicalDeviceFeatures deviceFeatures;
	deviceFeatures.setSamplerAnisotropy(vk::True);

	vk::DeviceCreateInfo createInfo({}, queueCreateInfos, {}, deviceExtensions, &deviceFeatures);

	if (appInfo::validation::enableValidationLayers)
	{
		createInfo.setPEnabledLayerNames(appInfo::validation::validationLayers);
	}

	return m_physicalDevice.createDevice(createInfo);
}