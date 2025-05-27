#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Window.hpp"

Window::Window(size_t width, size_t height, const char* title, vk::Instance instance) :
	m_width(width), m_height(height), m_title(title), m_instance(instance)
{
	createWindow();
	createSurface();
}

void Window::createWindow()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	m_window = glfwCreateWindow(m_width, m_height, m_title, nullptr, nullptr);
	// glfwSetWindowUserPointer(window, this);
	// glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void Window::createSurface()
{
	VkSurfaceKHR surface;
	if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
	m_surface = surface;
}

Window::~Window() noexcept
{
	m_instance.destroySurfaceKHR(m_surface);
	
	glfwDestroyWindow(m_window);
	glfwTerminate();
}