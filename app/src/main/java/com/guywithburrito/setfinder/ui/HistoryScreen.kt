package com.guywithburrito.setfinder.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.guywithburrito.setfinder.tracking.HistoryPersistence
import com.guywithburrito.setfinder.tracking.SavedSet
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun HistoryScreen(onBackClicked: () -> Unit) {
    val context = LocalContext.current
    val historyPersistence = remember { HistoryPersistence(context) }
    var history by remember { mutableStateOf(historyPersistence.loadHistory()) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Scan History") },
                navigationIcon = {
                    IconButton(onClick = onBackClicked) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = {
                        historyPersistence.clearHistory()
                        history = emptyList()
                    }) {
                        Icon(Icons.Default.DeleteSweep, contentDescription = "Clear All")
                    }
                },
                backgroundColor = MaterialTheme.colors.surface,
                contentColor = MaterialTheme.colors.onSurface
            )
        },
        backgroundColor = MaterialTheme.colors.background
    ) { padding ->
        if (history.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize().padding(padding), contentAlignment = Alignment.Center) {
                Text("No sets found yet. Start scanning!", style = MaterialTheme.typography.subtitle1)
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(padding),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                items(history) { savedSet ->
                    SavedSetItem(savedSet, historyPersistence)
                }
            }
        }
    }
}

@Composable
fun SavedSetItem(savedSet: SavedSet, persistence: HistoryPersistence) {
    val dateFormat = remember { SimpleDateFormat("MMM d, h:mm a", Locale.getDefault()) }
    
    val commonCount = remember(savedSet) { savedSet.cards.map { it.count }.distinct().size == 1 }
    val commonColor = remember(savedSet) { savedSet.cards.map { it.color }.distinct().size == 1 }
    val commonPattern = remember(savedSet) { savedSet.cards.map { it.pattern }.distinct().size == 1 }
    val commonShape = remember(savedSet) { savedSet.cards.map { it.shape }.distinct().size == 1 }

    Card(
        elevation = 4.dp,
        backgroundColor = MaterialTheme.colors.surface,
        contentColor = MaterialTheme.colors.onSurface
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = dateFormat.format(Date(savedSet.timestamp)),
                style = MaterialTheme.typography.caption,
                color = MaterialTheme.colors.onSurface.copy(alpha = 0.6f)
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                savedSet.cards.forEach { card ->
                    Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                        val bitmap = persistence.getCardBitmap(card.imageFilename)
                        if (bitmap != null) {
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = null,
                                modifier = Modifier.aspectRatio(0.7f).fillMaxWidth(),
                                contentScale = ContentScale.Crop
                            )
                        } else {
                            Box(modifier = Modifier.aspectRatio(0.7f).fillMaxWidth().background(MaterialTheme.colors.onSurface.copy(alpha = 0.1f)))
                        }
                        
                        Spacer(modifier = Modifier.height(4.dp))
                        
                        Text(
                            text = "${card.count} ${card.color}",
                            style = MaterialTheme.typography.overline,
                            fontWeight = if (commonCount || commonColor) FontWeight.ExtraBold else FontWeight.Normal
                        )
                        Text(
                            text = "${card.pattern} ${card.shape}",
                            style = MaterialTheme.typography.overline,
                            fontWeight = if (commonPattern || commonShape) FontWeight.ExtraBold else FontWeight.Normal
                        )
                    }
                }
            }
        }
    }
}
