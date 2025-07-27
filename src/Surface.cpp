#include "Window.hpp"

#include "Surface.hpp"

Surface::Surface(vk::Instance instance, const Window& window) :
	m_instance(instance), m_window(window), m_surface(m_window.createSurface(m_instance))
{}