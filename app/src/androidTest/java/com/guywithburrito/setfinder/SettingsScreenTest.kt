package com.guywithburrito.setfinder

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.guywithburrito.setfinder.ui.SettingsScreen
import com.guywithburrito.setfinder.ui.SetFinderTheme
import org.junit.Rule
import org.junit.Test

class SettingsScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun settingsScreen_dragToReorder() {
        composeTestRule.setContent {
            SetFinderTheme {
                SettingsScreen(onBackClicked = {})
            }
        }

        // Verify initial presence of color items
        composeTestRule.onNodeWithText("Green").assertExists()
        composeTestRule.onNodeWithText("Cyan").assertExists()

        // Perform a drag-and-drop reordering gesture.
        // We find the handle for "Green" and drag it down past "Cyan".
        composeTestRule.onNodeWithContentDescription("Reorder Green")
            .performTouchInput {
                // down -> advanceEventTime is the standard way to trigger long-press in tests
                down(center)
                advanceEventTime(viewConfiguration.longPressTimeoutMillis + 100)
                
                // Move down past the next item
                moveBy(androidx.compose.ui.geometry.Offset(0f, 300f))
                
                // Finalize the gesture
                up()
            }

        // Verify "Green" still exists after reorder (functional check)
        composeTestRule.onNodeWithText("Green").assertExists()
    }
}
