package com.guywithburrito.setfinder.tracking

import android.content.Context
import android.content.SharedPreferences
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb

class SettingsManager(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("set_finder_prefs", Context.MODE_PRIVATE)

    private val defaultColors = listOf(
        "Green" to Color.Green,
        "Cyan" to Color.Cyan,
        "Magenta" to Color.Magenta,
        "Yellow" to Color.Yellow,
        "Blue" to Color.Blue,
        "Orange" to Color(0xFFFFA500)
    )

    private val _highlightColors = mutableStateListOf<Pair<String, Color>>()
    val highlightColors: List<Pair<String, Color>> get() = _highlightColors

    init {
        loadColors()
    }

    private fun loadColors() {
        val savedColors = prefs.getString("highlight_colors", null)
        _highlightColors.clear()
        if (savedColors == null) {
            _highlightColors.addAll(defaultColors)
        } else {
            val parts = savedColors.split("|")
            parts.forEach { part ->
                val subParts = part.split(",")
                if (subParts.size == 2) {
                    try {
                        _highlightColors.add(subParts[0] to Color(subParts[1].toInt()))
                    } catch (e: Exception) {
                        // skip malformed
                    }
                }
            }
            if (_highlightColors.isEmpty()) _highlightColors.addAll(defaultColors)
        }
    }

    fun saveColors(colors: List<Pair<String, Color>>) {
        _highlightColors.clear()
        _highlightColors.addAll(colors)
        val serialized = colors.joinToString("|") { "${it.first},${it.second.toArgb()}" }
        prefs.edit().putString("highlight_colors", serialized).apply()
    }

    var sensitivity: Float
        get() = prefs.getFloat("sensitivity", 0.7f)
        set(value) = prefs.edit().putFloat("sensitivity", value).apply()

    var showLabels: Boolean
        get() = prefs.getBoolean("show_labels", true)
        set(value) = prefs.edit().putBoolean("show_labels", value).apply()

    var arMode: Boolean
        get() = prefs.getBoolean("ar_mode", false)
        set(value) = prefs.edit().putBoolean("ar_mode", value).apply()

    var singleCardMode: Boolean
        get() = prefs.getBoolean("single_card_mode", false)
        set(value) = prefs.edit().putBoolean("single_card_mode", value).apply()

    var useYOLO: Boolean
        get() = prefs.getBoolean("use_yolo", false)
        set(value) = prefs.edit().putBoolean("use_yolo", value).apply()
}
