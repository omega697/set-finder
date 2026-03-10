package com.guywithburrito.setfinder.ml

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.guywithburrito.setfinder.SetCardTestUtils
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import java.lang.StringBuilder

/**
 * Instrumented test for Stage 3 (Card Identification).
 * Uses the production CardExpert factory to ensure verification stays in sync with app logic.
 * This test is comprehensive and uses all meticulously arranged test chips in assets/chips/cards.
 */
@RunWith(AndroidJUnit4::class)
class CardExpertInstrumentationTest {

    @Test
    fun identify_isAccurateForAllTestChips() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        
        // Always test the production default as configured in the factory
        val expert = CardExpert.getInstance(appContext)
        val assets = testContext.assets.list("chips/cards") ?: emptyArray()
        
        val failureSummary = StringBuilder()
        var passed = 0
        var tested = 0
        
        assets.forEach { assetName ->
            if (!assetName.endsWith(".jpg") || assetName.contains("NONE")) return@forEach
            tested++
            
            val inputStream = testContext.assets.open("chips/cards/$assetName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            
            // Test the high-level identify API - exactly as the app uses it
            val actualCard = expert.identify(bitmap)
            
            val expectedCard = SetCardTestUtils.parseLabelFromFilename(assetName)
            val expectedStr = expectedCard?.let { SetCardTestUtils.formatLabel(it) } ?: "???"
            val actualStr = actualCard?.let { SetCardTestUtils.formatLabel(it) } ?: "NULL"
            
            if (actualStr == expectedStr) {
                passed++
            } else {
                failureSummary.append(String.format("  [%s]: expected [%s], but was [%s]\n", assetName, expectedStr, actualStr))
            }
        }
        
        expert.close()
        
        val accuracy = if (tested > 0) (passed.toFloat() / tested) else 1f
        android.util.Log.d("CardExpertTest", String.format("Accuracy: %.2f%% (%d/%d)", accuracy * 100, passed, tested))
        
        // We expect at least 98% accuracy on this verified set of chips
        if (accuracy < 0.98f) {
            val msg = String.format("Expert model accuracy too low: %.2f%% (%d/%d passed). Failures:\n%s", 
                                   accuracy * 100, passed, tested, failureSummary.toString())
            throw AssertionError(msg)
        }
    }
}
