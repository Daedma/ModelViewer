#include "Window.hpp"

#include "Surface.hpp"

Surface::Surface(vk::Instance instance, const Window& window) :
	m_instance(instance), m_surface()
{
	m_surface = window.createSurface(instance);
}