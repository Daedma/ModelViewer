#pragma once

#include <vulkan/vulkan.hpp>

#include "AppInfo.hpp"

class GLFWwindow;

class Window
{
public:
	Window(size_t width, size_t height, const char* title, vk::Instance instance);

	Window(vk::Instance instance) :
		Window(appInfo::general::windowSize.width, appInfo::general::windowSize.height, appInfo::general::appName, instance)
	{}

	~Window() noexcept;

	size_t getWidth() const noexcept { return m_width; }

	size_t getHeight() const noexcept { return m_height; }

	const char* getTitle() const noexcept { return m_title; }
private:
	GLFWwindow* m_window;
	size_t m_width;
	size_t m_height;
	const char* m_title;

	vk::Instance m_instance;

	vk::SurfaceKHR m_surface;

private:
	void createWindow();

	void createSurface();
};