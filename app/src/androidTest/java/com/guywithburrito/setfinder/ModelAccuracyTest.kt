package com.guywithburrito.setfinder

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import kotlin.test.assertNotNull

@RunWith(AndroidJUnit4::class)
class ModelAccuracyTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    data class TestChip(
        val asset: String,
        val count: SetCard.Count,
        val color: SetCard.Color,
        val pattern: SetCard.Pattern,
        val shape: SetCard.Shape
    )

    private val chips = listOf(
        TestChip("test_ONE_GREEN_EMPTY_DIAMOND.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.EMPTY, SetCard.Shape.DIAMOND),
        TestChip("test_ONE_GREEN_EMPTY_OVAL.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.EMPTY, SetCard.Shape.OVAL),
        TestChip("test_ONE_GREEN_EMPTY_SQUIGGLE.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.EMPTY, SetCard.Shape.SQUIGGLE),
        TestChip("test_ONE_GREEN_SHADED_DIAMOND.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.SHADED, SetCard.Shape.DIAMOND),
        TestChip("test_ONE_GREEN_SHADED_OVAL.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.SHADED, SetCard.Shape.OVAL),
        TestChip("test_ONE_GREEN_SHADED_SQUIGGLE.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.SHADED, SetCard.Shape.SQUIGGLE),
        TestChip("test_ONE_GREEN_SOLID_DIAMOND.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.SOLID, SetCard.Shape.DIAMOND),
        TestChip("test_ONE_GREEN_SOLID_OVAL.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.SOLID, SetCard.Shape.OVAL),
        TestChip("test_ONE_GREEN_SOLID_SQUIGGLE.jpg", SetCard.Count.ONE, SetCard.Color.GREEN, SetCard.Pattern.SOLID, SetCard.Shape.SQUIGGLE),
        TestChip("test_ONE_PURPLE_EMPTY_DIAMOND.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.EMPTY, SetCard.Shape.DIAMOND),
        TestChip("test_ONE_PURPLE_EMPTY_OVAL.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.EMPTY, SetCard.Shape.OVAL),
        TestChip("test_ONE_PURPLE_EMPTY_SQUIGGLE.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.EMPTY, SetCard.Shape.SQUIGGLE),
        TestChip("test_ONE_PURPLE_SHADED_DIAMOND.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.SHADED, SetCard.Shape.DIAMOND),
        TestChip("test_ONE_PURPLE_SHADED_OVAL.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.SHADED, SetCard.Shape.OVAL),
        TestChip("test_ONE_PURPLE_SHADED_SQUIGGLE.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.SHADED, SetCard.Shape.SQUIGGLE),
        TestChip("test_ONE_PURPLE_SOLID_DIAMOND.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.SOLID, SetCard.Shape.DIAMOND),
        TestChip("test_ONE_PURPLE_SOLID_OVAL.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.SOLID, SetCard.Shape.OVAL),
        TestChip("test_ONE_PURPLE_SOLID_SQUIGGLE.jpg", SetCard.Count.ONE, SetCard.Color.PURPLE, SetCard.Pattern.SOLID, SetCard.Shape.SQUIGGLE),
        TestChip("test_ONE_RED_EMPTY_DIAMOND.jpg", SetCard.Count.ONE, SetCard.Color.RED, SetCard.Pattern.EMPTY, SetCard.Shape.DIAMOND)
    )

    @Test
    fun testModelOnManyVerifiedChips() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val identifier = TFLiteCardIdentifier(appContext)

        var failures = 0
        val failureLog = StringBuilder()

        for (chip in chips) {
            try {
                val card = verifyCard(identifier, chip.asset)
                assertThat(card.count).isEqualTo(chip.count)
                assertThat(card.color).isEqualTo(chip.color)
                assertThat(card.pattern).isEqualTo(chip.pattern)
                assertThat(card.shape).isEqualTo(chip.shape)
            } catch (e: Throwable) {
                failures++
                failureLog.append("\nFAILED ${chip.asset}: ${e.message}")
            }
        }
        
        identifier.close()
        
        if (failures > 0) {
            org.junit.Assert.fail("Model accuracy test failed with $failures errors:$failureLog")
        }
    }

    private fun verifyCard(identifier: TFLiteCardIdentifier, assetName: String): SetCard {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        assertNotNull(bitmap, "Bitmap from $assetName should not be null")
        
        val card = identifier.identifyCard(bitmap)
        if (card == null) {
            android.util.Log.e("ModelTest", "FAILED TO IDENTIFY $assetName. Check logcat for 'TFLite' tags.")
        }
        assertNotNull(card, "Model returned null for $assetName")
        return card
    }
}
