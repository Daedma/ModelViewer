#pragma once

#include <cstdint>
#include <iostream>
#include <vulkan/vulkan.hpp>

namespace appInfo
{
	namespace general
	{

		constexpr const char* appName = "Model Viewer";

		constexpr struct
		{
			uint8_t major = 1;
			uint8_t minor = 0;
			uint8_t patch = 0;
		} appVersion;

		constexpr const char* engineName = "No Engine";

		constexpr struct
		{
			uint8_t major = 1;
			uint8_t minor = 0;
			uint8_t patch = 0;
		} engineVersion;

		constexpr struct
		{
			size_t width = 800;
			size_t height = 600;
		} windowSize;
	}

	namespace validation
	{
#ifdef NDEBUG
		constexpr bool enableValidationLayers = false;
#else
		constexpr bool enableValidationLayers = true;
#endif
		constexpr const char* validationLayers[] = { "VK_LAYER_KHRONOS_validation" };

		constexpr const char* debugMessengerExtension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

		constexpr PFN_vkDebugUtilsMessengerCallbackEXT debugCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData) -> VkBool32 {
				std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
				return VK_FALSE;
			};

		constexpr vk::DebugUtilsMessageSeverityFlagsEXT messageSeverity =
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;

		constexpr vk::DebugUtilsMessageTypeFlagsEXT messageType =
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
			vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
			vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

	}
}