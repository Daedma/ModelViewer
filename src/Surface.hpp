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

	const Window& getWindow() const noexcept { return m_window; }

private:
	vk::Instance m_instance;

	const Window& m_window;

	vk::SurfaceKHR m_surface;
};