package com.guywithburrito.setfinder.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.guywithburrito.setfinder.card.SetCard
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

data class SavedSet(
    @SerializedName("id") val id: String = UUID.randomUUID().toString(),
    @SerializedName("timestamp") val timestamp: Long = System.currentTimeMillis(),
    @SerializedName("cards") val cards: List<SavedSetCard>
) {
    fun calculateFingerprint(): String {
        return cards.map { "${it.shape}|${it.pattern}|${it.count}|${it.color}" }
            .sorted()
            .joinToString("||")
    }
}

data class SavedSetCard(
    @SerializedName("shape") val shape: SetCard.Shape,
    @SerializedName("pattern") val pattern: SetCard.Pattern,
    @SerializedName("count") val count: SetCard.Count,
    @SerializedName("color") val color: SetCard.Color,
    @SerializedName("imageFilename") val imageFilename: String
)

class HistoryPersistence(private val context: Context) {
    private val prefs = context.getSharedPreferences("history_prefs", Context.MODE_PRIVATE)
    private val gson = Gson()
    private val historyDir = File(context.filesDir, "set_history").apply { if (!exists()) mkdirs() }

    fun saveSet(cards: List<Pair<SetCard, Bitmap>>) {
        val savedCards = cards.map { (card, bitmap) ->
            val filename = "card_${UUID.randomUUID()}.png"
            val file = File(historyDir, filename)
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            SavedSetCard(card.shape, card.pattern, card.count, card.color, filename)
        }
        
        val newSet = SavedSet(cards = savedCards)
        val currentHistory = loadHistory().toMutableList()
        currentHistory.add(0, newSet)
        
        // Keep only last 50 sets
        val trimmedHistory = currentHistory.take(50)
        prefs.edit().putString("history_json", gson.toJson(trimmedHistory)).apply()
        
        // Cleanup old images
        cleanupOrphanedImages(trimmedHistory)
    }

    fun loadHistory(): List<SavedSet> {
        val json = prefs.getString("history_json", null) ?: return emptyList()
        return try {
            gson.fromJson(json, Array<SavedSet>::class.java).toList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun getExistingFingerprints(): Set<String> {
        return loadHistory().map { it.calculateFingerprint() }.toSet()
    }

    fun getCardBitmap(filename: String): Bitmap? {
        val file = File(historyDir, filename)
        return if (file.exists()) BitmapFactory.decodeFile(file.absolutePath) else null
    }

    fun clearHistory() {
        prefs.edit().clear().apply()
        historyDir.deleteRecursively()
        historyDir.mkdirs()
    }

    private fun cleanupOrphanedImages(history: List<SavedSet>) {
        val activeFiles = history.flatMap { s -> s.cards.map { it.imageFilename } }.toSet()
        historyDir.listFiles()?.forEach { file ->
            if (file.name !in activeFiles) {
                file.delete()
            }
        }
    }
}
