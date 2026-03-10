package com.guywithburrito.setfinder.tracking

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import com.guywithburrito.setfinder.card.SetCard
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

/**
 * This test evaluates the HistoryPersistence class, verifying that detected 
 * sets and their associated card images are correctly saved to and loaded 
 * from device storage, including history limits and clearing logic.
 */
@RunWith(RobolectricTestRunner::class)
class HistoryPersistenceTest {

    private lateinit var context: Context
    private lateinit var persistence: HistoryPersistence

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        persistence = HistoryPersistence(context)
        persistence.clearHistory()
    }

    @Test
    fun saveSet_persistsDataAndImages() {
        val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val card = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        
        val cards = listOf(card to bitmap, card to bitmap, card to bitmap)
        persistence.saveSet(cards)

        val history = persistence.loadHistory()
        assertThat(history).hasSize(1)
        assertThat(history[0].cards).hasSize(3)
        
        val filename = history[0].cards[0].imageFilename
        val imageFile = File(context.filesDir, "set_history/$filename")
        assertThat(imageFile.exists()).isTrue()
        
        val loadedBitmap = persistence.getCardBitmap(filename)
        assertThat(loadedBitmap).isNotNull()
    }

    @Test
    fun limitHistory_keepsOnlyLatest() {
        val bitmap = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
        val card = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        val cards = listOf(card to bitmap, card to bitmap, card to bitmap)

        // Save 60 sets (limit is 50)
        repeat(60) {
            persistence.saveSet(cards)
        }

        val history = persistence.loadHistory()
        assertThat(history).hasSize(50)
    }

    @Test
    fun clearHistory_removesAll() {
        val bitmap = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
        val card = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        persistence.saveSet(listOf(card to bitmap, card to bitmap, card to bitmap))

        persistence.clearHistory()
        assertThat(persistence.loadHistory()).isEmpty()
        
        val historyDir = File(context.filesDir, "set_history")
        assertThat(historyDir.listFiles()).isEmpty()
    }
}
