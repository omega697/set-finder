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
        var passed = 0
        var tested = 0
        
        // 1. Test Confirmed Cards (in chips/ directory)
        testContext.assets.list("chips")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            tested++
            val inputStream = testContext.assets.open("chips/$assetName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            val isCard = filter.isCard(bitmap)
            if (isCard) passed++
            logBuilder.append(String.format("chips/%s: isCard=%b -> %s\n", assetName, isCard, if (isCard) "PASS" else "FAIL"))
        }

        // 2. Test Confirmed Non-Cards
        testContext.assets.list("non_cards")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            tested++
            val inputStream = testContext.assets.open("non_cards/$assetName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            val isCard = filter.isCard(bitmap)
            if (!isCard) passed++
            logBuilder.append(String.format("non_cards/%s: isCard=%b -> %s\n", assetName, isCard, if (!isCard) "PASS" else "FAIL"))
        }
        
        filter.close()
        
        android.util.Log.d("CardFilterStageTest", "Results:\n$logBuilder")
        android.util.Log.d("CardFilterStageTest", "Passed $passed / $tested")
        
        // Check if we hit at least 90% accuracy on this diverse set (the goal is 100%, but 0.1 threshold might be tricky)
        assertThat(passed).isEqualTo(tested)
    }
}
