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
		m_instance(create(config)), m_enableValidationLayers(config.isValidationEnabled())
	{
		if (m_enableValidationLayers)
		{
			vk::DebugUtilsMessengerCreateInfoEXT createInfo(
				{},
				appInfo::validation::messageSeverity,
				appInfo::validation::messageType,
				config.debugCallback);
			vk::DispatchLoaderDynamic dldi(m_instance, vkGetInstanceProcAddr);
			m_debugMessenger = m_instance.createDebugUtilsMessengerEXT(createInfo, nullptr, dldi);
		}
	}

	~Instance() noexcept
	{
		if (m_enableValidationLayers)
		{
			vk::DispatchLoaderDynamic dldi{ m_instance, vkGetInstanceProcAddr };
			m_instance.destroyDebugUtilsMessengerEXT(m_debugMessenger, nullptr, dldi);
		}
		m_instance.destroy();
	};

	Instance(const Instance&) = delete;
	Instance& operator=(const Instance&) = delete;

	vk::Instance get() const noexcept { return m_instance; }
private:
	vk::Instance m_instance;
	vk::PhysicalDevice m_physicalDevice; // TODO: do not forget to initialize this

	bool m_enableValidationLayers;
	vk::DebugUtilsMessengerEXT m_debugMessenger;

private:

	vk::Instance create(const InstanceConfig& config);

	static bool checkValidationLayerSupport(const std::vector<const char*>& validationLayers);
};