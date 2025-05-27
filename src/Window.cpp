#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Window.hpp"

Window::Window(size_t width, size_t height, const char* title) :
	m_width(width), m_height(height), m_title(title)
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	m_window = glfwCreateWindow(m_width, m_height, m_title, nullptr, nullptr);
	// glfwSetWindowUserPointer(window, this);
	// glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

Window::~Window() noexcept
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
}