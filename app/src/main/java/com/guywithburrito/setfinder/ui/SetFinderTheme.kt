package com.guywithburrito.setfinder.ui

import androidx.compose.material.MaterialTheme
import androidx.compose.material.lightColors
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val SetFinderColors = lightColors(
    primary = Color(0xFF1976D2),
    primaryVariant = Color(0xFF115293),
    secondary = Color(0xFF4CAF50),
    background = Color.White,
    surface = Color.White,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onBackground = Color.Black,
    onSurface = Color.Black
)

@Composable
fun SetFinderTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colors = SetFinderColors,
        content = content
    )
}
