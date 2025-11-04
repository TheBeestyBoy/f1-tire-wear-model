#!/usr/bin/env python3
"""
Download 15 classic books with similar style to Bible for training data.
These books are from Project Gutenberg and have formal, literary language.
"""

import urllib.request
import os
import time

def clean_gutenberg_text(text):
    """Remove Project Gutenberg header and footer."""
    # Remove header
    if "*** START OF" in text:
        text = text.split("*** START OF")[1]
        # Remove the project gutenberg line that follows
        if "***" in text:
            text = text.split("***", 1)[1]

    # Remove footer
    if "*** END OF" in text:
        text = text.split("*** END OF")[0]

    return text.strip()

def download_book(url, output_path, book_name):
    """Download and clean a single book."""
    try:
        print(f"  Downloading {book_name}...")
        urllib.request.urlretrieve(url, output_path)

        # Clean the file
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text = clean_gutenberg_text(text)

        # Save cleaned version
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        chars = len(text)
        words = len(text.split())
        print(f"    ✓ {book_name}: {chars:,} chars, {words:,} words")
        return True

    except Exception as e:
        print(f"    ✗ Failed to download {book_name}: {e}")
        return False

def download_training_books():
    """Download 45 classic books with formal, literary style."""

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Books with similar formal/archaic style to Bible
    books = [
        # More Shakespeare (already have 5, adding 8 more)
        ("https://www.gutenberg.org/files/1539/1539-0.txt", "data/othello.txt", "Othello by Shakespeare"),
        ("https://www.gutenberg.org/files/1540/1540-0.txt", "data/merchant_venice.txt", "Merchant of Venice by Shakespeare"),
        ("https://www.gutenberg.org/files/1120/1120-0.txt", "data/midsummer_nights_dream.txt", "Midsummer Night's Dream by Shakespeare"),
        ("https://www.gutenberg.org/files/1041/1041-0.txt", "data/tempest.txt", "The Tempest by Shakespeare"),
        ("https://www.gutenberg.org/files/1129/1129-0.txt", "data/twelfth_night.txt", "Twelfth Night by Shakespeare"),
        ("https://www.gutenberg.org/files/1777/1777-0.txt", "data/as_you_like_it.txt", "As You Like It by Shakespeare"),
        ("https://www.gutenberg.org/files/1793/1793-0.txt", "data/much_ado.txt", "Much Ado About Nothing by Shakespeare"),
        ("https://www.gutenberg.org/files/1534/1534-0.txt", "data/richard_iii.txt", "Richard III by Shakespeare"),

        # Classical Greek literature
        ("https://www.gutenberg.org/files/1727/1727-0.txt", "data/odyssey.txt", "The Odyssey by Homer"),
        ("https://www.gutenberg.org/files/6130/6130-0.txt", "data/iliad.txt", "The Iliad by Homer"),
        ("https://www.gutenberg.org/files/1656/1656-0.txt", "data/aeneid.txt", "The Aeneid by Virgil"),
        ("https://www.gutenberg.org/files/31/31-0.txt", "data/apology_plato.txt", "Apology by Plato"),
        ("https://www.gutenberg.org/files/1658/1658-0.txt", "data/symposium_plato.txt", "Symposium by Plato"),

        # Religious and philosophical texts
        ("https://www.gutenberg.org/files/3296/3296-0.txt", "data/koran.txt", "The Koran"),
        ("https://www.gutenberg.org/files/5200/5200-0.txt", "data/metamorphoses.txt", "Metamorphoses by Ovid"),
        ("https://www.gutenberg.org/files/3600/3600-0.txt", "data/essays_bacon.txt", "Essays by Francis Bacon"),
        ("https://www.gutenberg.org/files/4363/4363-0.txt", "data/book_mormon.txt", "Book of Mormon"),
        ("https://www.gutenberg.org/files/25525/25525-0.txt", "data/summa_theologica.txt", "Summa Theologica by Aquinas"),

        # Medieval and Renaissance literature
        ("https://www.gutenberg.org/files/2701/2701-0.txt", "data/moby_dick.txt", "Moby Dick by Melville"),
        ("https://www.gutenberg.org/files/2554/2554-0.txt", "data/crime_punishment.txt", "Crime and Punishment by Dostoevsky"),
        ("https://www.gutenberg.org/files/1399/1399-0.txt", "data/anna_karenina.txt", "Anna Karenina by Tolstoy"),
        ("https://www.gutenberg.org/files/2600/2600-0.txt", "data/war_peace.txt", "War and Peace by Tolstoy"),
        ("https://www.gutenberg.org/files/1342/1342-0.txt", "data/pride_prejudice.txt", "Pride and Prejudice by Austen"),

        # More classic literature with formal language
        ("https://www.gutenberg.org/files/730/730-0.txt", "data/oliver_twist.txt", "Oliver Twist by Dickens"),
        ("https://www.gutenberg.org/files/98/98-0.txt", "data/tale_two_cities.txt", "A Tale of Two Cities by Dickens"),
        ("https://www.gutenberg.org/files/1400/1400-0.txt", "data/great_expectations.txt", "Great Expectations by Dickens"),
        ("https://www.gutenberg.org/files/766/766-0.txt", "data/david_copperfield.txt", "David Copperfield by Dickens"),

        # Epic poetry and classical works
        ("https://www.gutenberg.org/files/26/26-0.txt", "data/leaves_of_grass.txt", "Leaves of Grass by Whitman"),
        ("https://www.gutenberg.org/files/1184/1184-0.txt", "data/count_monte_cristo.txt", "Count of Monte Cristo by Dumas"),
        ("https://www.gutenberg.org/files/219/219-0.txt", "data/heart_of_darkness.txt", "Heart of Darkness by Conrad"),
        ("https://www.gutenberg.org/files/84/84-0.txt", "data/frankenstein.txt", "Frankenstein by Shelley"),
        ("https://www.gutenberg.org/files/2641/2641-0.txt", "data/brothers_karamazov.txt", "Brothers Karamazov by Dostoevsky"),

        # Historical and philosophical
        ("https://www.gutenberg.org/files/1998/1998-0.txt", "data/thus_spoke_zarathustra.txt", "Thus Spoke Zarathustra by Nietzsche"),
        ("https://www.gutenberg.org/files/2500/2500-0.txt", "data/meditations_marcus.txt", "Meditations by Marcus Aurelius"),
        ("https://www.gutenberg.org/files/4300/4300-0.txt", "data/ulysses.txt", "Ulysses by Joyce"),
        ("https://www.gutenberg.org/files/345/345-0.txt", "data/dracula.txt", "Dracula by Stoker"),
        ("https://www.gutenberg.org/files/174/174-0.txt", "data/picture_dorian_gray.txt", "Picture of Dorian Gray by Wilde"),

        # Additional epic and mythological works
        ("https://www.gutenberg.org/files/1661/1661-0.txt", "data/sherlock_holmes.txt", "Adventures of Sherlock Holmes by Doyle"),
        ("https://www.gutenberg.org/files/1260/1260-0.txt", "data/jane_eyre.txt", "Jane Eyre by Brontë"),
        ("https://www.gutenberg.org/files/768/768-0.txt", "data/wuthering_heights.txt", "Wuthering Heights by Brontë"),
        ("https://www.gutenberg.org/files/158/158-0.txt", "data/emma.txt", "Emma by Austen"),
        ("https://www.gutenberg.org/files/1259/1259-0.txt", "data/twenty_thousand_leagues.txt", "Twenty Thousand Leagues by Verne"),
        ("https://www.gutenberg.org/files/76/76-0.txt", "data/huckleberry_finn.txt", "Huckleberry Finn by Twain"),
        ("https://www.gutenberg.org/files/74/74-0.txt", "data/tom_sawyer.txt", "Tom Sawyer by Twain"),

        # === ADDING 100 NEW BOOKS ===

        # More Classic Literature
        ("https://www.gutenberg.org/files/11/11-0.txt", "data/alice_wonderland.txt", "Alice in Wonderland by Carroll"),
        ("https://www.gutenberg.org/files/215/215-0.txt", "data/call_of_wild.txt", "Call of the Wild by London"),
        ("https://www.gutenberg.org/files/1232/1232-0.txt", "data/prince.txt", "The Prince by Machiavelli"),
        ("https://www.gutenberg.org/files/4280/4280-0.txt", "data/les_miserables.txt", "Les Misérables by Hugo"),
        ("https://www.gutenberg.org/files/1661/1661-0.txt", "data/sherlock_study.txt", "Sherlock: Study in Scarlet by Doyle"),

        # Jane Austen Collection
        ("https://www.gutenberg.org/files/105/105-0.txt", "data/persuasion.txt", "Persuasion by Austen"),
        ("https://www.gutenberg.org/files/121/121-0.txt", "data/northanger_abbey.txt", "Northanger Abbey by Austen"),
        ("https://www.gutenberg.org/files/141/141-0.txt", "data/mansfield_park.txt", "Mansfield Park by Austen"),
        ("https://www.gutenberg.org/files/946/946-0.txt", "data/lady_susan.txt", "Lady Susan by Austen"),

        # Charles Dickens Collection
        ("https://www.gutenberg.org/files/46/46-0.txt", "data/christmas_carol.txt", "A Christmas Carol by Dickens"),
        ("https://www.gutenberg.org/files/786/786-0.txt", "data/bleak_house.txt", "Bleak House by Dickens"),
        ("https://www.gutenberg.org/files/580/580-0.txt", "data/little_dorrit.txt", "Little Dorrit by Dickens"),
        ("https://www.gutenberg.org/files/883/883-0.txt", "data/our_mutual_friend.txt", "Our Mutual Friend by Dickens"),
        ("https://www.gutenberg.org/files/967/967-0.txt", "data/hard_times.txt", "Hard Times by Dickens"),

        # Mark Twain Collection
        ("https://www.gutenberg.org/files/86/86-0.txt", "data/prince_and_pauper.txt", "Prince and Pauper by Twain"),
        ("https://www.gutenberg.org/files/119/119-0.txt", "data/yankee_court.txt", "Connecticut Yankee by Twain"),
        ("https://www.gutenberg.org/files/245/245-0.txt", "data/innocents_abroad.txt", "Innocents Abroad by Twain"),
        ("https://www.gutenberg.org/files/3177/3177-0.txt", "data/puddnhead_wilson.txt", "Pudd'nhead Wilson by Twain"),

        # H.G. Wells
        ("https://www.gutenberg.org/files/35/35-0.txt", "data/time_machine.txt", "Time Machine by Wells"),
        ("https://www.gutenberg.org/files/36/36-0.txt", "data/war_worlds.txt", "War of the Worlds by Wells"),
        ("https://www.gutenberg.org/files/5230/5230-0.txt", "data/invisible_man.txt", "Invisible Man by Wells"),
        ("https://www.gutenberg.org/files/159/159-0.txt", "data/island_dr_moreau.txt", "Island of Dr. Moreau by Wells"),

        # Jules Verne
        ("https://www.gutenberg.org/files/103/103-0.txt", "data/around_world_80_days.txt", "Around World in 80 Days by Verne"),
        ("https://www.gutenberg.org/files/164/164-0.txt", "data/journey_center_earth.txt", "Journey to Center of Earth by Verne"),
        ("https://www.gutenberg.org/files/1268/1268-0.txt", "data/mysterious_island.txt", "Mysterious Island by Verne"),
        ("https://www.gutenberg.org/files/83/83-0.txt", "data/from_earth_moon.txt", "From Earth to Moon by Verne"),

        # More Russian Classics
        ("https://www.gutenberg.org/files/2638/2638-0.txt", "data/idiot.txt", "The Idiot by Dostoevsky"),
        ("https://www.gutenberg.org/files/8117/8117-0.txt", "data/demons.txt", "Demons by Dostoevsky"),
        ("https://www.gutenberg.org/files/600/600-0.txt", "data/notes_underground.txt", "Notes from Underground by Dostoevsky"),
        ("https://www.gutenberg.org/files/985/985-0.txt", "data/war_peace_tolstoy.txt", "War and Peace Vol 1 by Tolstoy"),
        ("https://www.gutenberg.org/files/2142/2142-0.txt", "data/resurrection.txt", "Resurrection by Tolstoy"),

        # Edgar Allan Poe
        ("https://www.gutenberg.org/files/2147/2147-0.txt", "data/raven.txt", "The Raven by Poe"),
        ("https://www.gutenberg.org/files/1063/1063-0.txt", "data/cask_amontillado.txt", "Cask of Amontillado by Poe"),
        ("https://www.gutenberg.org/files/2148/2148-0.txt", "data/masque_red_death.txt", "Masque of Red Death by Poe"),
        ("https://www.gutenberg.org/files/932/932-0.txt", "data/fall_house_usher.txt", "Fall of House of Usher by Poe"),

        # Oscar Wilde
        ("https://www.gutenberg.org/files/790/790-0.txt", "data/importance_earnest.txt", "Importance of Being Earnest by Wilde"),
        ("https://www.gutenberg.org/files/773/773-0.txt", "data/happy_prince.txt", "Happy Prince by Wilde"),
        ("https://www.gutenberg.org/files/902/902-0.txt", "data/canterville_ghost.txt", "Canterville Ghost by Wilde"),

        # Arthur Conan Doyle - Sherlock Holmes
        ("https://www.gutenberg.org/files/244/244-0.txt", "data/sign_of_four.txt", "Sign of Four by Doyle"),
        ("https://www.gutenberg.org/files/2097/2097-0.txt", "data/hound_baskervilles.txt", "Hound of Baskervilles by Doyle"),
        ("https://www.gutenberg.org/files/834/834-0.txt", "data/valley_of_fear.txt", "Valley of Fear by Doyle"),
        ("https://www.gutenberg.org/files/2350/2350-0.txt", "data/memoirs_sherlock.txt", "Memoirs of Sherlock Holmes by Doyle"),

        # More Shakespeare
        ("https://www.gutenberg.org/files/1795/1795-0.txt", "data/king_lear.txt", "King Lear by Shakespeare"),
        ("https://www.gutenberg.org/files/2250/2250-0.txt", "data/henry_iv_part1.txt", "Henry IV Part 1 by Shakespeare"),
        ("https://www.gutenberg.org/files/1536/1536-0.txt", "data/henry_v.txt", "Henry V by Shakespeare"),
        ("https://www.gutenberg.org/files/2267/2267-0.txt", "data/henry_vi_part1.txt", "Henry VI Part 1 by Shakespeare"),
        ("https://www.gutenberg.org/files/1532/1532-0.txt", "data/henry_viii.txt", "Henry VIII by Shakespeare"),

        # Greek Classics
        ("https://www.gutenberg.org/files/1726/1726-0.txt", "data/oedipus_king.txt", "Oedipus the King by Sophocles"),
        ("https://www.gutenberg.org/files/1725/1725-0.txt", "data/antigone.txt", "Antigone by Sophocles"),
        ("https://www.gutenberg.org/files/2696/2696-0.txt", "data/agamemnon.txt", "Agamemnon by Aeschylus"),
        ("https://www.gutenberg.org/files/8264/8264-0.txt", "data/prometheus_bound.txt", "Prometheus Bound by Aeschylus"),
        ("https://www.gutenberg.org/files/2397/2397-0.txt", "data/medea.txt", "Medea by Euripides"),

        # More Epic Poetry
        ("https://www.gutenberg.org/files/20/20-0.txt", "data/paradise_lost.txt", "Paradise Lost by Milton"),
        ("https://www.gutenberg.org/files/58/58-0.txt", "data/paradise_regained.txt", "Paradise Regained by Milton"),
        ("https://www.gutenberg.org/files/3346/3346-0.txt", "data/rime_ancient_mariner.txt", "Rime of Ancient Mariner by Coleridge"),
        ("https://www.gutenberg.org/files/1245/1245-0.txt", "data/divine_comedy.txt", "Divine Comedy by Dante"),

        # Philosophy
        ("https://www.gutenberg.org/files/1497/1497-0.txt", "data/republic.txt", "The Republic by Plato"),
        ("https://www.gutenberg.org/files/1656/1656-0.txt", "data/phaedo.txt", "Phaedo by Plato"),
        ("https://www.gutenberg.org/files/1600/1600-0.txt", "data/discourse_method.txt", "Discourse on Method by Descartes"),
        ("https://www.gutenberg.org/files/5827/5827-0.txt", "data/tractatus.txt", "Tractatus by Spinoza"),
        ("https://www.gutenberg.org/files/4705/4705-0.txt", "data/critique_pure_reason.txt", "Critique of Pure Reason by Kant"),

        # More Gothic/Horror
        ("https://www.gutenberg.org/files/42324/42324-0.txt", "data/jekyll_and_hyde.txt", "Jekyll and Hyde by Stevenson"),
        ("https://www.gutenberg.org/files/345/345-0.txt", "data/dracula_full.txt", "Dracula Full by Stoker"),
        ("https://www.gutenberg.org/files/41/41-0.txt", "data/legend_sleepy_hollow.txt", "Legend of Sleepy Hollow by Irving"),

        # American Classics
        ("https://www.gutenberg.org/files/16/16-0.txt", "data/peter_pan.txt", "Peter Pan by Barrie"),
        ("https://www.gutenberg.org/files/17/17-0.txt", "data/tale_of_genji.txt", "Tale of Genji"),
        ("https://www.gutenberg.org/files/205/205-0.txt", "data/walden.txt", "Walden by Thoreau"),
        ("https://www.gutenberg.org/files/2680/2680-0.txt", "data/meditations_descartes.txt", "Meditations by Descartes"),
        ("https://www.gutenberg.org/files/1400/1400-0.txt", "data/great_gatsby.txt", "Great Gatsby by Fitzgerald"),

        # Victorian Literature
        ("https://www.gutenberg.org/files/161/161-0.txt", "data/sense_sensibility.txt", "Sense and Sensibility by Austen"),
        ("https://www.gutenberg.org/files/1400/1400-0.txt", "data/middlemarch.txt", "Middlemarch by Eliot"),
        ("https://www.gutenberg.org/files/550/550-0.txt", "data/scarlet_letter.txt", "Scarlet Letter by Hawthorne"),
        ("https://www.gutenberg.org/files/77/77-0.txt", "data/house_seven_gables.txt", "House of Seven Gables by Hawthorne"),

        # Adventure Novels
        ("https://www.gutenberg.org/files/120/120-0.txt", "data/treasure_island.txt", "Treasure Island by Stevenson"),
        ("https://www.gutenberg.org/files/209/209-0.txt", "data/kidnapped.txt", "Kidnapped by Stevenson"),
        ("https://www.gutenberg.org/files/700/700-0.txt", "data/last_mohicans.txt", "Last of Mohicans by Cooper"),
        ("https://www.gutenberg.org/files/1155/1155-0.txt", "data/ivanhoe.txt", "Ivanhoe by Scott"),
        ("https://www.gutenberg.org/files/136/136-0.txt", "data/little_women.txt", "Little Women by Alcott"),

        # Historical Fiction
        ("https://www.gutenberg.org/files/33/33-0.txt", "data/scarlet_pimpernel.txt", "Scarlet Pimpernel by Orczy"),
        ("https://www.gutenberg.org/files/1257/1257-0.txt", "data/three_musketeers.txt", "Three Musketeers by Dumas"),
        ("https://www.gutenberg.org/files/1259/1259-0.txt", "data/man_iron_mask.txt", "Man in Iron Mask by Dumas"),

        # Satire and Social Commentary
        ("https://www.gutenberg.org/files/829/829-0.txt", "data/gullivers_travels.txt", "Gulliver's Travels by Swift"),
        ("https://www.gutenberg.org/files/113/113-0.txt", "data/secret_garden.txt", "Secret Garden by Burnett"),
        ("https://www.gutenberg.org/files/145/145-0.txt", "data/middlemarch_eliot.txt", "Middlemarch by George Eliot"),

        # Modern Classics
        ("https://www.gutenberg.org/files/996/996-0.txt", "data/don_quixote.txt", "Don Quixote by Cervantes"),
        ("https://www.gutenberg.org/files/2591/2591-0.txt", "data/grimms_fairy_tales.txt", "Grimm's Fairy Tales"),
        ("https://www.gutenberg.org/files/1952/1952-0.txt", "data/yellow_wallpaper.txt", "Yellow Wallpaper by Gilman"),
        ("https://www.gutenberg.org/files/514/514-0.txt", "data/little_prince.txt", "Little Prince by Saint-Exupéry"),

        # Science and Natural Philosophy
        ("https://www.gutenberg.org/files/2009/2009-0.txt", "data/origin_species.txt", "Origin of Species by Darwin"),
        ("https://www.gutenberg.org/files/1952/1952-0.txt", "data/common_sense.txt", "Common Sense by Paine"),
        ("https://www.gutenberg.org/files/27942/27942-0.txt", "data/leviathan.txt", "Leviathan by Hobbes"),

        # Romance
        ("https://www.gutenberg.org/files/398/398-0.txt", "data/tess_durbervilles.txt", "Tess of d'Urbervilles by Hardy"),
        ("https://www.gutenberg.org/files/110/110-0.txt", "data/taming_of_shrew.txt", "Taming of Shrew by Shakespeare"),
        ("https://www.gutenberg.org/files/1260/1260-0.txt", "data/villette.txt", "Villette by Brontë"),
    ]

    print("="*70)
    print("DOWNLOADING 145 CLASSIC BOOKS FOR TRAINING")
    print("="*70)
    print("These books include classical literature, religious texts,")
    print("Shakespeare, epic poetry, and philosophical works.\n")

    success_count = 0
    total_chars = 0
    total_words = 0

    for i, (url, output_path, book_name) in enumerate(books, 1):
        print(f"\n[{i}/145] {book_name}")

        if download_book(url, output_path, book_name):
            success_count += 1

            # Get stats
            with open(output_path, 'r', encoding='utf-8') as f:
                text = f.read()
                total_chars += len(text)
                total_words += len(text.split())

        # Be nice to Project Gutenberg servers
        time.sleep(1)

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Successfully downloaded: {success_count}/{len(books)} books")
    print(f"Total characters: {total_chars:,}")
    print(f"Total words: {total_words:,}")
    print(f"Total size: {total_chars/1024/1024:.2f} MB")
    print()
    print("NEXT STEPS:")
    print("  1. Run: python scripts/combine_bibles.py")
    print("     (This will combine all texts into training_text.txt)")
    print("  2. Run: python app/train_hybrid.py")
    print("     (This will train with the hybrid configuration)")
    print("="*70)

    return success_count > 0

if __name__ == "__main__":
    # Change to backend directory if running from project root
    if os.path.basename(os.getcwd()) != "backend":
        if os.path.exists("backend"):
            os.chdir("backend")

    download_training_books()
