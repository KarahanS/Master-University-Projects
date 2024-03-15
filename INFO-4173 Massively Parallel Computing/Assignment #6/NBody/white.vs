#version 140
in vec2 inPosition;

void main(void)
{
	// Transformation to view space
	gl_Position = vec4(inPosition, 0.0, 1.0);
}
