#pragma once

#include <vulkan/vulkan.hpp>

#include "AppInfo.hpp"

class Window;

class Surface
{
public:
	Surface(vk::Instance instance, const Window& window);

	~Surface() noexcept
	{
		m_instance.destroySurfaceKHR(m_surface);
	}

	vk::SurfaceKHR get() const noexcept { return m_surface; }

private:
	vk::Instance m_instance;

	vk::SurfaceKHR m_surface;
};