package com.guywithburrito.setfinder.ml

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import java.lang.StringBuilder

@RunWith(AndroidJUnit4::class)
class CardFilterInstrumentationTest {

    @Test
    fun isCard_correctlyClassifiesChips() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        
        // Stage 2: Modular Filter
        val filter = CardFilter.getInstance(appContext)
        
        val logBuilder = StringBuilder()
        val failures = StringBuilder()
        var passed = 0
        var tested = 0
        
        // 1. Test Confirmed Cards
        testContext.assets.list("chips/cards")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            tested++
            val inputStream = testContext.assets.open("chips/cards/$assetName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            val isCard = filter.isCard(bitmap)
            
            if (isCard) {
                passed++
            } else {
                failures.append(String.format("  [chips/cards/%s]: expected isCard=true, but was false\n", assetName))
            }
            logBuilder.append(String.format("chips/cards/%s: isCard=%b -> %s\n", assetName, isCard, if (isCard) "PASS" else "FAIL"))
        }

        // 2. Test Confirmed Non-Cards
        testContext.assets.list("chips/non_cards")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            tested++
            val inputStream = testContext.assets.open("chips/non_cards/$assetName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            val isCard = filter.isCard(bitmap)
            
            if (!isCard) {
                passed++
            } else {
                failures.append(String.format("  [chips/non_cards/%s]: expected isCard=false, but was true\n", assetName))
            }
            logBuilder.append(String.format("chips/non_cards/%s: isCard=%b -> %s\n", assetName, isCard, if (!isCard) "PASS" else "FAIL"))
        }
        
        filter.close()
        
        val accuracy = if (tested > 0) (passed.toFloat() / tested) else 1f
        android.util.Log.d("CardFilterStageTest", String.format("Accuracy: %.2f%% (%d/%d)", accuracy * 100, passed, tested))
        
        // We expect at least 98% accuracy for Stage 2 Filtering
        if (accuracy < 0.98f) {
            val msg = String.format("Card filter accuracy too low: %.2f%% (%d/%d passed). Failures:\n%s", 
                                   accuracy * 100, passed, tested, failures.toString())
            throw AssertionError(msg)
        }
    }
}
