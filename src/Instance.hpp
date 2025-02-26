#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <vector>
#include <string>

#include "AppInfo.hpp"

struct InstanceConfig
{
	std::string appName;
	uint32_t appVersion;
	std::string engineName;
	uint32_t engineVersion;
	std::vector<const char*> extensions;
	std::vector<const char*> layers;
	PFN_vkDebugUtilsMessengerCallbackEXT debugCallback;

	bool isValidationEnabled() const noexcept { return !layers.empty(); }

	static InstanceConfig getDefaultConfig(bool enableValidationLayers);
};

class Instance
{
public:
	Instance(const InstanceConfig& config = InstanceConfig::getDefaultConfig(appInfo::validation::enableValidationLayers)) :
		instance(create(config)), enableValidationLayers(config.isValidationEnabled())
	{
		if (enableValidationLayers)
		{
			vk::DebugUtilsMessengerCreateInfoEXT createInfo(
				{},
				appInfo::validation::messageSeverity,
				appInfo::validation::messageType,
				config.debugCallback);
			vk::DispatchLoaderDynamic dldi(instance, vkGetInstanceProcAddr);
			debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo, nullptr, dldi);
		}
	}

	~Instance() noexcept
	{
		if (enableValidationLayers)
		{
			vk::DispatchLoaderDynamic dldi{ instance, vkGetInstanceProcAddr };
			instance.destroyDebugUtilsMessengerEXT(debugMessenger, nullptr, dldi);
		}
		instance.destroy();
	};

	Instance(const Instance&) = delete;
	Instance& operator=(const Instance&) = delete;

	vk::Instance get() const noexcept { return instance; }
private:
	vk::Instance instance;
	vk::PhysicalDevice physicalDevice;

	bool enableValidationLayers;
	vk::DebugUtilsMessengerEXT debugMessenger;

private:

	vk::Instance create(const InstanceConfig& config);

	static bool checkValidationLayerSupport(const std::vector<const char*>& validationLayers);
};