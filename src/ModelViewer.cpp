#include <vulkan/vulkan.hpp>

#include <iterator>

#include "ModelViewer.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

void ModelViewer::createInstance()
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

	instance = vk::createInstance(createInfo);
}

bool ModelViewer::checkValidationLayerSupport()
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

std::vector<const char*> ModelViewer::getRequiredExtensions()
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

vk::DebugUtilsMessengerCreateInfoEXT ModelViewer::getDebugMessengerCreateInfo()
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

void ModelViewer::setupDebugMessenger()
{
	if (!enableValidationLayers) return;
	VkDebugUtilsMessengerCreateInfoEXT createInfo = getDebugMessengerCreateInfo();
	vk::DispatchLoaderDynamic dldi(instance, vkGetInstanceProcAddr);
	debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo, nullptr, dldi);
}

void ModelViewer::createSurface()
{
	VkSurfaceKHR surface;
	if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
	this->surface = surface;
}

void ModelViewer::pickPhysicalDevice()
{
	auto devices = instance.enumeratePhysicalDevices();
	if (devices.empty())
	{
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}

	for (const auto& device : devices)
	{
		if (isDeviceSuitable(device))
		{
			physicalDevice = device;
			msaaSamples = getMaxUsableSampleCount();
			break;
		}
	}

	if (!physicalDevice)
	{
		throw std::runtime_error("failed to find a suitable GPU!");
	}
}

bool ModelViewer::isDeviceSuitable(vk::PhysicalDevice device)
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

QueueFamilyIndices ModelViewer::findQueueFamilies(vk::PhysicalDevice device)
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

		vk::Bool32 presentSupport = device.getSurfaceSupportKHR(i, surface);

		if (presentSupport)
		{
			indices.presentFamily = i;
		}

		if (indices.isComplete())
		{
			break;
		}

		i++;
	}

	return indices;
}

bool ModelViewer::checkDeviceExtensionSupport(vk::PhysicalDevice device)
{
	auto availableExtensions = device.enumerateDeviceExtensionProperties();
	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	for (const auto& extension : availableExtensions)
	{
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

SwapChainSupportDetails ModelViewer::querySwapChainSupport(vk::PhysicalDevice device)
{
	SwapChainSupportDetails details;
	details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
	details.formats = device.getSurfaceFormatsKHR(surface);
	details.presentModes = device.getSurfacePresentModesKHR(surface);
	return details;
}

void ModelViewer::createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

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

	if (enableValidationLayers)
	{
		createInfo.setPEnabledLayerNames(validationLayers);
	}

	device = physicalDevice.createDevice(createInfo);

	graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
	presentQueue = device.getQueue(indices.presentFamily.value(), 0);
}

void ModelViewer::createSwapChain()
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

	vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 &&
		imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR createInfo({},
		surface,
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

	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	std::array<uint32_t, 2> queueFamilyIndices{ indices.graphicsFamily.value(), indices.presentFamily.value() };

	if (indices.graphicsFamily != indices.presentFamily)
	{
		createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
		createInfo.setQueueFamilyIndices(queueFamilyIndices);
	}

	swapChain = device.createSwapchainKHR(createInfo);
	swapChainImages = device.getSwapchainImagesKHR(swapChain);
	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;
}

vk::SurfaceFormatKHR ModelViewer::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	for (const auto& availableFormat : availableFormats)
	{
		if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
			availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
		{
			return availableFormat;
		}
	}

	return availableFormats[0];
}

vk::PresentModeKHR ModelViewer::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
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

vk::Extent2D ModelViewer::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return capabilities.currentExtent;
	}
	else
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		vk::Extent2D actualExtent(width, height);

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}

void ModelViewer::createImageViews()
{
	swapChainImageViews.resize(swapChainImages.size());
	std::transform(swapChainImages.begin(), swapChainImages.end(), swapChainImageViews.begin(),
		[&](vk::Image image) {
			return createImageView(image, swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);
		});
}

vk::ImageView ModelViewer::createImageView(vk::Image image, vk::Format format,
	vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
{
	vk::ImageViewCreateInfo viewInfo(
		{},
		image,
		vk::ImageViewType::e2D,
		format,
		{},
		{ aspectFlags, 0, mipLevels, 0, 1 });
	return device.createImageView(viewInfo);
}

void ModelViewer::createRenderPass()
{
	vk::AttachmentDescription colorAttachment(
		{},
		swapChainImageFormat,
		msaaSamples,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal);
	vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

	vk::AttachmentDescription colorAttachmentResolve(
		{},
		swapChainImageFormat,
		vk::SampleCountFlagBits::e1,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::ePresentSrcKHR
	);
	vk::AttachmentReference colorAttachmentResolveRef(2, vk::ImageLayout::eColorAttachmentOptimal);

	vk::AttachmentDescription depthAttachment(
		{},
		findDepthFormat(),
		msaaSamples,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eDontCare,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthStencilAttachmentOptimal);
	vk::AttachmentReference depthAttachmentRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

	vk::SubpassDescription subpass(
		{},
		vk::PipelineBindPoint::eGraphics,
		{},
		colorAttachmentRef,
		colorAttachmentResolveRef,
		&depthAttachmentRef,
		{}
	);

	vk::SubpassDependency dependency(
		vk::SubpassExternal,
		0,
		vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
		vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
		{},
		vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
		{}
	);

	std::array<vk::AttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
	vk::RenderPassCreateInfo renderPassInfo(
		{},
		attachments,
		subpass,
		dependency
	);

	renderPass = device.createRenderPass(renderPassInfo);
}

vk::Format ModelViewer::findDepthFormat()
{
	return findSupportedFormat(
		{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
		vk::ImageTiling::eOptimal,
		vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

vk::Format ModelViewer::findSupportedFormat(const std::vector<vk::Format>& candidates,
	vk::ImageTiling tiling, vk::FormatFeatureFlags features)
{
	for (vk::Format format : candidates)
	{
		vk::FormatProperties props = physicalDevice.getFormatProperties(format);
		if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
		{
			return format;
		}
		else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
		{
			return format;
		}
	}

	throw std::runtime_error("failed to find supported format!");
}

void ModelViewer::createDescriptorSetLayout()
{
	vk::DescriptorSetLayoutBinding uboLayoutBinding(
		0,
		vk::DescriptorType::eUniformBuffer,
		1,
		vk::ShaderStageFlagBits::eVertex,
		nullptr
	);

	vk::DescriptorSetLayoutBinding samplerLayoutBinding(
		1,
		vk::DescriptorType::eCombinedImageSampler,
		1,
		vk::ShaderStageFlagBits::eFragment,
		nullptr
	);

	std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
	vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
	descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
}

void ModelViewer::createGraphicsPipeline()
{
	auto vertShaderCode = readFile("shaders/vert.spv");
	auto fragShaderCode = readFile("shaders/frag.spv");

	vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo(
		{},
		vk::ShaderStageFlagBits::eVertex,
		vertShaderModule,
		"main"
	);

	vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
		{},
		vk::ShaderStageFlagBits::eFragment,
		fragShaderModule,
		"main"
	);

	std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{ vertShaderStageInfo, fragShaderStageInfo };

	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
		{},
		bindingDescription,
		attributeDescriptions
	);

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, vk::False);

	vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);

	vk::PipelineRasterizationStateCreateInfo rasterizer(
		{},
		vk::False,
		vk::False,
		vk::PolygonMode::eFill,
		vk::CullModeFlagBits::eBack,
		vk::FrontFace::eCounterClockwise,
		vk::False
	);
	rasterizer.setLineWidth(1.f);

	vk::PipelineMultisampleStateCreateInfo multisampling({}, msaaSamples, vk::False);

	vk::PipelineColorBlendAttachmentState colorBlendAttachment;
	colorBlendAttachment
		.setBlendEnable(vk::False)
		.setColorWriteMask(
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA
		);

	vk::PipelineColorBlendStateCreateInfo colorBlending(
		{},
		vk::False,
		vk::LogicOp::eCopy,
		colorBlendAttachment,
		{ 0.f }
	);

	std::vector<vk::DynamicState> dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};
	vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

	vk::PipelineDepthStencilStateCreateInfo depthStencil(
		{},
		vk::True,
		vk::True,
		vk::CompareOp::eLess,
		vk::False,
		vk::False,
		{},
		{},
		0.f,
		1.f
	);

	pipelineLayout = device.createPipelineLayout({ {}, descriptorSetLayout });

	vk::GraphicsPipelineCreateInfo pipelineInfo{
		{},
		shaderStages,
		&vertexInputInfo,
		&inputAssembly,
		nullptr,
		&viewportState,
		&rasterizer,
		&multisampling,
		&depthStencil,
		&colorBlending,
		&dynamicState,
		pipelineLayout,
		renderPass,
		0,
		{},
		0
	};

	auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);

	if (result.result != vk::Result::eSuccess)
	{
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	graphicsPipeline = result.value;

	device.destroyShaderModule(fragShaderModule);
	device.destroyShaderModule(vertShaderModule);
}

vk::ShaderModule ModelViewer::createShaderModule(const std::vector<uint32_t>& code)
{
	return device.createShaderModule({ {}, code });
}

void ModelViewer::createCommandPool()
{
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

	vk::CommandPoolCreateInfo poolInfo{
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		queueFamilyIndices.graphicsFamily.value()
	};

	commandPool = device.createCommandPool(poolInfo);
}

void ModelViewer::createColorResources()
{
	vk::Format colorFormat = swapChainImageFormat;

	createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat,
		vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
		vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage, colorImageMemory);
	colorImageView = createImageView(colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
}

void ModelViewer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples,
	vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
	vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory)
{
	vk::ImageCreateInfo imageInfo{
		{},
		vk::ImageType::e2D,
		format,
		{ width, height, 1 },
		mipLevels,
		1,
		numSamples,
		tiling,
		usage,
		vk::SharingMode::eExclusive,
		nullptr,
		vk::ImageLayout::eUndefined
	};

	image = device.createImage(imageInfo);

	vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);

	vk::MemoryAllocateInfo allocInfo(
		memRequirements.size,
		findMemoryType(memRequirements.memoryTypeBits, properties)
	);

	imageMemory = device.allocateMemory(allocInfo);
	device.bindImageMemory(image, imageMemory, 0);
}

uint32_t ModelViewer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
	vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

	for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

void ModelViewer::createDepthResources()
{
	vk::Format depthFormat = findDepthFormat();
	createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
	depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

void ModelViewer::createFramebuffers()
{
	swapChainFramebuffers.resize(swapChainImageViews.size());

	for (size_t i = 0; i < swapChainImageViews.size(); i++)
	{
		std::array<vk::ImageView, 3> attachments = {
			colorImageView,
			depthImageView,
			swapChainImageViews[i]
		};

		vk::FramebufferCreateInfo framebufferInfo{
			{},
			renderPass,
			attachments,
			swapChainExtent.width,
			swapChainExtent.height,
			1
		};

		swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
	}
}

void ModelViewer::createTextureImage()
{
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	vk::DeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	void* data = device.mapMemory(stagingBufferMemory, 0, imageSize);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	device.unmapMemory(stagingBufferMemory);

	stbi_image_free(pixels);

	createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
		vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

	transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined,
		vk::ImageLayout::eTransferDstOptimal, mipLevels);
	copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	//transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

	generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);

	device.destroyBuffer(stagingBuffer);
	device.freeMemory(stagingBufferMemory);
}

void ModelViewer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory)
{
	vk::BufferCreateInfo bufferInfo{
		{},
		size,
		usage,
		vk::SharingMode::eExclusive,
		{}
	};

	buffer = device.createBuffer(bufferInfo);

	vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);

	vk::MemoryAllocateInfo allocInfo{
		memRequirements.size,
		findMemoryType(memRequirements.memoryTypeBits, properties)
	};

	bufferMemory = device.allocateMemory(allocInfo);
	device.bindBufferMemory(buffer, bufferMemory, 0);
}

void ModelViewer::transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels)
{
	vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

	vk::ImageMemoryBarrier barrier{
		{},
		{},
		oldLayout,
		newLayout,
		vk::QueueFamilyIgnored,
		vk::QueueFamilyIgnored,
		image,
		{ vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1 }
	};

	vk::PipelineStageFlags sourceStage;
	vk::PipelineStageFlags destinationStage;

	if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
	{
		barrier.setSrcAccessMask({});
		barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

		sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage = vk::PipelineStageFlagBits::eTransfer;
	}
	else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
		newLayout == vk::ImageLayout::eTransferDstOptimal)
	{
		barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
		barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

		sourceStage = vk::PipelineStageFlagBits::eTransfer;
		destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
	}
	else
	{
		throw std::invalid_argument("unsupported layout transition!");
	}

	commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, barrier);
	endSingleTimeCommands(commandBuffer);
}

vk::CommandBuffer ModelViewer::beginSingleTimeCommands()
{
	vk::CommandBufferAllocateInfo allocInfo{ commandPool, vk::CommandBufferLevel::ePrimary, 1 };

	vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

	commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

	return commandBuffer;
}

void ModelViewer::endSingleTimeCommands(vk::CommandBuffer commandBuffer)
{
	commandBuffer.end();

	vk::SubmitInfo submitInfo{ {}, {}, commandBuffer, {} };

	graphicsQueue.submit(submitInfo);
	graphicsQueue.waitIdle();

	device.freeCommandBuffers(commandPool, commandBuffer);
}

void ModelViewer::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
{
	vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

	vk::BufferImageCopy region{
		0,
		0,
		0,
		{ vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
		{ 0, 0, 0 },
		{ width, height, 1 }
	};

	commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

	endSingleTimeCommands(commandBuffer);
}

void ModelViewer::generateMipmaps(vk::Image image, vk::Format imageFormat, int32_t texWidth,
	int32_t texHeight, uint32_t mipLevels)
{
	// Check if image format supports linear blitting
	vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
	if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
	{
		throw std::runtime_error("texture image format does not support linear blitting!");
	}

	vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

	vk::ImageMemoryBarrier barrier{
		{},
		{},
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eUndefined,
		vk::QueueFamilyIgnored,
		vk::QueueFamilyIgnored,
		image,
		{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
	};

	int32_t mipWidth = texWidth;
	int32_t mipHeight = texHeight;

	for (uint32_t i = 1; i != mipLevels; ++i)
	{
		barrier.subresourceRange.setBaseMipLevel(i - 1);
		barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
		barrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
		barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
		barrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, barrier);

		vk::ImageBlit blit{
			{ vk::ImageAspectFlagBits::eColor, i - 1, 0, 1 },
			{ { { 0, 0, 0 }, { mipWidth, mipHeight, 1 } } },
			{ vk::ImageAspectFlagBits::eColor, i, 0, 1 },
			{ { { 0, 0, 0 }, { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 } } }
		};

		commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
			image, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

		barrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
		barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferRead);
		barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, barrier);

		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;
	}


	barrier.subresourceRange.setBaseMipLevel(mipLevels - 1);
	barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
	barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
	barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
	barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, barrier);

	endSingleTimeCommands(commandBuffer);
}

void ModelViewer::createTextureImageView()
{
	textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb,
		vk::ImageAspectFlagBits::eColor, mipLevels);
}

void ModelViewer::createTextureSampler()
{
	vk::SamplerCreateInfo samplerInfo{
		{},
		vk::Filter::eLinear,
		vk::Filter::eLinear,
		vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat,
		vk::SamplerAddressMode::eRepeat,
		vk::SamplerAddressMode::eRepeat,
		0.f,
		vk::True,
		physicalDevice.getProperties().limits.maxSamplerAnisotropy,
		vk::False,
		vk::CompareOp::eAlways,
		0.f,
		static_cast<float>(mipLevels),
		vk::BorderColor::eFloatTransparentBlack,
		vk::False
	};

	textureSampler = device.createSampler(samplerInfo);
}

void ModelViewer::loadModel()
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
	{
		throw std::runtime_error(warn + err);
	}

	std::unordered_map<Vertex, uint32_t> uniqueVertices{};

	for (const auto& shape : shapes)
	{
		for (const auto& index : shape.mesh.indices)
		{
			Vertex vertex{};

			vertex.pos = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.texCoord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};

			vertex.color = { 1.0f, 1.0f, 1.0f };

			if (uniqueVertices.count(vertex) == 0)
			{
				uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
				vertices.push_back(vertex);
			}

			indices.push_back(uniqueVertices[vertex]);
		}
	}
}

void ModelViewer::createVertexBuffer()
{
	vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
	memcpy(data, vertices.data(), bufferSize);
	device.unmapMemory(stagingBufferMemory);

	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

	copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

	device.destroyBuffer(stagingBuffer);
	device.freeMemory(stagingBufferMemory);
}

void ModelViewer::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
	vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

	commandBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy{ 0, 0, size });

	endSingleTimeCommands(commandBuffer);
}

void ModelViewer::createIndexBuffer()
{
	vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

	vk::Buffer stagingBuffer;
	vk::DeviceMemory stagingBufferMemory;
	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
	memcpy(data, indices.data(), bufferSize);
	device.unmapMemory(stagingBufferMemory);

	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

	copyBuffer(stagingBuffer, indexBuffer, bufferSize);

	device.destroyBuffer(stagingBuffer);
	device.freeMemory(stagingBufferMemory);
}

void ModelViewer::createUniformBuffers()
{
	vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

	uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i)
	{
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible,
			uniformBuffers[i], uniformBuffersMemory[i]);

		uniformBuffersMapped[i] = device.mapMemory(uniformBuffersMemory[i], 0, bufferSize, {});
	}
}

void ModelViewer::createDescriptorPool()
{
	std::array<vk::DescriptorPoolSize, 2> poolSizes{ {
		{ vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) },
		{ vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) }
		} };

	vk::DescriptorPoolCreateInfo poolInfo{
		{},
		static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
		poolSizes
	};

	descriptorPool = device.createDescriptorPool(poolInfo);
}

void ModelViewer::createDescriptorSets()
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, layouts };

	descriptorSets = device.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i)
	{
		vk::DescriptorBufferInfo bufferInfo{ uniformBuffers[i], 0, sizeof(UniformBufferObject) };

		vk::DescriptorImageInfo imageInfo{
			textureSampler,
			textureImageView,
			vk::ImageLayout::eShaderReadOnlyOptimal
		};

		std::array<vk::WriteDescriptorSet, 2> descriptorWrites;

		descriptorWrites[0]
			.setDstSet(descriptorSets[i])
			.setDstBinding(0)
			.setDstArrayElement(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setBufferInfo(bufferInfo);

		descriptorWrites[1]
			.setDstSet(descriptorSets[i])
			.setDstBinding(1)
			.setDstArrayElement(0)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setImageInfo(imageInfo);

		device.updateDescriptorSets(descriptorWrites, nullptr);
	}
}

void ModelViewer::createCommandBuffers()
{
	commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

	vk::CommandBufferAllocateInfo allocInfo{
		commandPool,
		vk::CommandBufferLevel::ePrimary,
		MAX_FRAMES_IN_FLIGHT
	};

	commandBuffers = device.allocateCommandBuffers(allocInfo);
}

void ModelViewer::createSyncObjects()
{
	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i)
	{
		imageAvailableSemaphores[i] = device.createSemaphore({});
		renderFinishedSemaphores[i] = device.createSemaphore({});
		inFlightFences[i] = device.createFence({ vk::FenceCreateFlagBits::eSignaled });
	}
}

void ModelViewer::drawFrame()
{
	vk::Result waitForFencesResult = device.waitForFences(inFlightFences[currentFrame], vk::True, UINT64_MAX);

	if (waitForFencesResult != vk::Result::eSuccess)
	{
		throw std::runtime_error{ "failed to wait for fences!" };
	}

	uint32_t imageIndex;
	auto acquireResult = device.acquireNextImageKHR(swapChain, UINT64_MAX,
		imageAvailableSemaphores[currentFrame], nullptr);

	switch (acquireResult.result)
	{
	case vk::Result::eErrorOutOfDateKHR:
		recreateSwapChain();
		return;
	case vk::Result::eSuccess:
	case vk::Result::eSuboptimalKHR:
		imageIndex = acquireResult.value;
		break;
	default:
		throw std::runtime_error("failed to acquire swap chain image!");
		break;
	}

	device.resetFences(inFlightFences[currentFrame]);

	commandBuffers[currentFrame].reset();
	recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

	updateUniformBuffer(currentFrame);

	vk::PipelineStageFlags waitStages{ vk::PipelineStageFlagBits::eColorAttachmentOutput };

	vk::SubmitInfo submitInfo{
		imageAvailableSemaphores[currentFrame],
		waitStages,
		commandBuffers[currentFrame],
		renderFinishedSemaphores[currentFrame]
	};

	graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

	vk::PresentInfoKHR presentInfo{
		renderFinishedSemaphores[currentFrame],
		swapChain,
		imageIndex,
		{}
	};

	vk::Result presentResult = presentQueue.presentKHR(presentInfo);

	if (presentResult == vk::Result::eErrorOutOfDateKHR ||
		presentResult == vk::Result::eSuboptimalKHR || framebufferResized)
	{
		framebufferResized = false;
		recreateSwapChain();
	}
	else if (presentResult != vk::Result::eSuccess)
	{
		throw std::runtime_error("failed to present swap chain image!");
	}

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void ModelViewer::recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex)
{
	commandBuffer.begin(vk::CommandBufferBeginInfo{});

	std::array<vk::ClearValue, 2> clearValues{};
	clearValues[0].setColor({ 0.0f, 0.0f, 0.0f, 1.0f });
	clearValues[1].setDepthStencil({ 1.0f, 0 });

	vk::RenderPassBeginInfo renderPassInfo{
		renderPass,
		swapChainFramebuffers[imageIndex],
		{ { 0, 0 }, swapChainExtent },
		clearValues
	};

	commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

	vk::Viewport viewport{
		0.f,
		0.f,
		static_cast<float>(swapChainExtent.width),
		static_cast<float>(swapChainExtent.height),
		0.f,
		1.f
	};
	commandBuffer.setViewport(0, viewport);

	vk::Rect2D scissor{ { 0, 0 }, swapChainExtent };
	commandBuffer.setScissor(0, scissor);

	commandBuffer.bindVertexBuffers(0, vertexBuffer, { 0 });

	commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
		pipelineLayout, 0, descriptorSets[currentFrame], nullptr);

	commandBuffer.drawIndexed(indices.size(), 1, 0, 0, 0);

	commandBuffer.endRenderPass();

	commandBuffer.end();
}

void ModelViewer::updateUniformBuffer(uint32_t currentImage)
{
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	UniformBufferObject ubo{};
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
	ubo.proj[1][1] *= -1;

	memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void ModelViewer::cleanupSwapChain()
{
	device.destroyImageView(colorImageView);
	device.destroyImage(colorImage);
	device.freeMemory(colorImageMemory);

	for (auto framebuffer : swapChainFramebuffers)
	{
		device.destroyFramebuffer(framebuffer);
	}

	for (auto imageView : swapChainImageViews)
	{
		device.destroyImageView(imageView);
	}

	device.destroySwapchainKHR(swapChain);
}

void ModelViewer::recreateSwapChain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	device.waitIdle();

	cleanupSwapChain();

	createSwapChain();
	createImageViews();
	createColorResources();
	createDepthResources();
	createFramebuffers();
}

void ModelViewer::cleanup()
{
	cleanupSwapChain();

	device.destroyImageView(depthImageView);
	device.destroyImage(depthImage);
	device.freeMemory(depthImageMemory);

	device.destroySampler(textureSampler);
	device.destroyImageView(textureImageView);
	device.destroyImage(textureImage);
	device.freeMemory(textureImageMemory);

	for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i)
	{
		device.destroyBuffer(uniformBuffers[i]);
		device.freeMemory(uniformBuffersMemory[i]);
	}

	device.destroyDescriptorPool(descriptorPool);
	device.destroyDescriptorSetLayout(descriptorSetLayout);
	device.destroyPipeline(graphicsPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyRenderPass(renderPass);

	device.destroyBuffer(indexBuffer);
	device.freeMemory(indexBufferMemory);

	device.destroyBuffer(vertexBuffer);
	device.freeMemory(vertexBufferMemory);

	for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i)
	{
		device.destroySemaphore(renderFinishedSemaphores[i]);
		device.destroySemaphore(imageAvailableSemaphores[i]);
		device.destroyFence(inFlightFences[i]);
	}

	device.destroyCommandPool(commandPool);

	device.destroy();

	if (enableValidationLayers)
	{
		vk::DispatchLoaderDynamic dldi{ instance, vkGetInstanceProcAddr };
		instance.destroyDebugUtilsMessengerEXT(debugMessenger, nullptr, dldi);
	}

	instance.destroySurfaceKHR(surface);

	instance.destroy();

	glfwDestroyWindow(window);

	glfwTerminate();
}