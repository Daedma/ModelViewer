#include <GLFW/glfw3.h>

#include <iostream>
#include <iterator>

#include "Instance.hpp"
#include "AppInfo.hpp"

InstanceConfig InstanceConfig::getDefaultConfig(bool enableValidationLayers)
{
	InstanceConfig config{};
	config.appName = appInfo::general::appName;
	config.appVersion = VK_MAKE_VERSION(
		appInfo::general::appVersion.major,
		appInfo::general::appVersion.minor,
		appInfo::general::appVersion.patch);
	config.engineName = appInfo::general::engineName;
	config.engineVersion = VK_MAKE_VERSION(
		appInfo::general::engineVersion.major,
		appInfo::general::engineVersion.minor,
		appInfo::general::engineVersion.patch);

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	config.extensions.assign(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers)
	{
		config.extensions.emplace_back(appInfo::validation::debugMessengerExtension);
		config.layers.assign(std::cbegin(appInfo::validation::validationLayers),
			std::cend(appInfo::validation::validationLayers));
		config.debugCallback = appInfo::validation::debugCallback;
	}

	return config;
}

vk::Instance Instance::create(const InstanceConfig& config)
{
	if (config.isValidationEnabled() && !checkValidationLayerSupport(config.layers))
	{
		throw std::runtime_error("validation layers requested, but not available!");
	}

	vk::ApplicationInfo appInfo(
		config.appName.c_str(),
		config.appVersion,
		config.engineName.c_str(),
		config.engineVersion);

	vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo(
		{},
		appInfo::validation::messageSeverity,
		appInfo::validation::messageType,
		config.debugCallback);

	vk::InstanceCreateInfo createInfo(
		{},
		&appInfo,
		config.layers,
		config.extensions,
		config.isValidationEnabled() ? &debugCreateInfo : nullptr);

	return vk::createInstance(createInfo);
}

bool Instance::checkValidationLayerSupport(const std::vector<const char*>& validationLayers)
{
	auto availableLayers = vk::enumerateInstanceLayerProperties();

	for (const char* layerName : validationLayers)
	{
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers)
		{
			if (strcmp(layerName, layerProperties.layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
		{
			return false;
		}
	}

	return true;
}
