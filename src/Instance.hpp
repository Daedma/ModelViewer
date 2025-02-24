#pragma once

#include <vulkan/vulkan.hpp>

#include <array>

class Instance
{
public:
	Instance(vk::InstanceCreateInfo createInfo_ = getDefaultCreateInfo()) :
		instance(vk::createInstance(createInfo_))
	{
		if (enableValidationLayers)
		{
			VkDebugUtilsMessengerCreateInfoEXT createInfo = getDebugMessengerCreateInfo();
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
	vk::DebugUtilsMessengerEXT debugMessenger;

	static vk::InstanceCreateInfo getDefaultCreateInfo();

	static constexpr std::array<const char*, 1> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	static bool checkValidationLayerSupport();

	static std::vector<const char*> getRequiredExtensions();

	static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo() noexcept;

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData);

#ifdef NDEBUG
	static constexpr bool enableValidationLayers = false;
#else
	static constexpr bool enableValidationLayers = true;
#endif
};