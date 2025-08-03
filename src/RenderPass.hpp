#pragma once

#include <vulkan/vulkan.hpp>

class Device;

class RenderPass
{
public:
	RenderPass(const Device& device);

	~RenderPass() noexcept;

	vk::RenderPass get() const noexcept { return m_renderPass; }
private:
	const Device& m_device;
	vk::RenderPass m_renderPass;
};