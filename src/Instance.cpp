#include <GLFW/glfw3.h>

#include <iostream>

#include "Instance.hpp"


vk::InstanceCreateInfo Instance::getDefaultCreateInfo()
{
	if (enableValidationLayers && !checkValidationLayerSupport())
	{
		throw std::runtime_error("validation layers requested, but not available!");
	}

	vk::ApplicationInfo appInfo("Model Viewer",
		VK_MAKE_VERSION(1, 0, 0),
		"No Engine",
		VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_1);

	auto extensions = getRequiredExtensions();
	vk::InstanceCreateInfo createInfo({}, &appInfo, {}, extensions);

	vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo = getDebugMessengerCreateInfo();
	if (enableValidationLayers)
	{
		createInfo.setPEnabledLayerNames(validationLayers);
		createInfo.setPNext(&debugCreateInfo);
	}

	return createInfo;
}

bool Instance::checkValidationLayerSupport()
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

std::vector<const char*> Instance::getRequiredExtensions()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers)
	{
		extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Instance::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT Instance::getDebugMessengerCreateInfo() noexcept
{
	vk::DebugUtilsMessengerCreateInfoEXT createInfo;
	createInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	createInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
	createInfo.setPfnUserCallback(debugCallback);
	return createInfo;
}

