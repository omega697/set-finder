package com.guywithburrito.setfinder.tracking

import android.content.Context
import androidx.compose.ui.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * This test evaluates the SettingsManager, ensuring that user preferences for 
 * highlight colors, sensitivity, and labels are correctly persisted and 
 * retrieved using Robolectric-backed SharedPreferences.
 */
@RunWith(RobolectricTestRunner::class)
class SettingsManagerTest {

    private lateinit var context: Context
    private lateinit var settingsManager: SettingsManager

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        settingsManager = SettingsManager(context)
    }

    @Test
    fun defaultSettings_areCorrect() {
        assertThat(settingsManager.highlightColors).hasSize(6)
        assertThat(settingsManager.sensitivity).isEqualTo(0.7f)
        assertThat(settingsManager.showLabels).isTrue()
    }

    @Test
    fun saveColors_persistsCorrectly() {
        val newColors = listOf(
            "Cyan" to Color.Cyan,
            "Green" to Color.Green
        )
        settingsManager.saveColors(newColors)

        // Create a new instance to verify persistence
        val newManager = SettingsManager(context)
        assertThat(newManager.highlightColors).hasSize(2)
        assertThat(newManager.highlightColors[0].first).isEqualTo("Cyan")
        assertThat(newManager.highlightColors[0].second).isEqualTo(Color.Cyan)
    }

    @Test
    fun sensitivity_persistsCorrectly() {
        settingsManager.sensitivity = 0.9f
        
        val newManager = SettingsManager(context)
        assertThat(newManager.sensitivity).isEqualTo(0.9f)
    }

    @Test
    fun showLabels_persistsCorrectly() {
        settingsManager.showLabels = false
        
        val newManager = SettingsManager(context)
        assertThat(newManager.showLabels).isFalse()
    }
}
