#version 330 core
layout (location = 0) in vec3 in_position;
uniform mat4 viewProj;

void main() {
    gl_Position = viewProj * vec4(in_position, 1.0);
}