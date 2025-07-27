#pragma once

#include <vulkan/vulkan.hpp>

#include "AppInfo.hpp"

class GLFWwindow;

class Window
{
public:
	using NativeWindowHandle = GLFWwindow*;

public:
	Window(size_t width, size_t height, const char* title);

	Window() :
		Window(appInfo::general::windowSize.width, appInfo::general::windowSize.height, appInfo::general::appName)
	{}

	~Window() noexcept;

	vk::SurfaceKHR createSurface(const vk::Instance& instance) const;

	vk::Extent2D getFramebufferExtent() const;

	NativeWindowHandle get() const noexcept { return m_window; }

	size_t getWidth() const noexcept { return m_width; }

	size_t getHeight() const noexcept { return m_height; }

	const char* getTitle() const noexcept { return m_title; }
private:
	NativeWindowHandle m_window;
	size_t m_width;
	size_t m_height;
	const char* m_title;
};