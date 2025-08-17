import torch
import torch.nn.functional as F
import networkx as nx
from graphgen.dfscode.dfs_wrapper import graph_from_dfscode


def compute_loss(pred, targets, pad_token_id):
    loss_total = 0.0

    weights = {
        'v1': 3.0,
        'v2': 3.0,
        'l1': 1.0,
        'e':  1.0,
        'l2': 1.0,
    }


    # Token-level cross entropy losses
    for key in targets:
        logits = pred[key]
        target = targets[key]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=pad_token_id
        )
        loss_total += loss * weights[key]

    return loss_total 



def evaluate(model, val_loader, pad_token_id, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            logits = model(inputs)

            loss = compute_loss(logits, targets, pad_token_id)

            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_model(model, train_dataloader, val_dataloader, optimizer, device, pad_token_id, model_name, num_epochs=1000):
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            optimizer.zero_grad()
            logits = model(inputs)
            loss = compute_loss(logits, targets, pad_token_id)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        val_loss = evaluate(model, val_dataloader, pad_token_id, device)
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def generate_dfs_code(
    model,
    node_id_vocab,
    node_label_vocab,
    edge_label_vocab,
    max_len=64,
    temperature=1.0,
    device='cuda'
):
    model.eval()
    model.to(device)

    pad_id = node_id_vocab.get_id("<PAD>")
    end_id = node_id_vocab.get_id("<END>")
    start_token = ("<START>", "<START>", "<START>", "<START>", "<START>")

    # Prepare first token
    v1_seq = [node_id_vocab.get_id(start_token[0])]
    v2_seq = [node_id_vocab.get_id(start_token[1])]
    l1_seq = [node_label_vocab.get_id(start_token[2])]
    e_seq  = [edge_label_vocab.get_id(start_token[3])]
    l2_seq = [node_label_vocab.get_id(start_token[4])]

    for _ in range(max_len - 1):  # Already have one token
        # Prepare inputs
        inputs = {
            'v1': torch.tensor([v1_seq], dtype=torch.long, device=device),
            'v2': torch.tensor([v2_seq], dtype=torch.long, device=device),
            'l1': torch.tensor([l1_seq], dtype=torch.long, device=device),
            'e':  torch.tensor([e_seq],  dtype=torch.long, device=device),
            'l2': torch.tensor([l2_seq], dtype=torch.long, device=device),
        }

        # Forward pass
        with torch.no_grad():
            logits = model(inputs)

        next_vals = {}
        for key in logits:
            last_logits = logits[key][:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            idx = torch.multinomial(probs, 1)
            next_vals[key] = idx.item()

        # Append predicted token
        v1_seq.append(next_vals['v1'])
        v2_seq.append(next_vals['v2'])
        l1_seq.append(next_vals['l1'])
        e_seq.append(next_vals['e'])
        l2_seq.append(next_vals['l2'])

        # Check if we reached the end token
        if any(next_vals[k] == end_id for k in ['v1', 'v2', 'l1', 'e', 'l2']):
            break

        # Optional stopping condition
        if any(next_vals[k] == pad_id for k in ['v1', 'v2', 'l1', 'e', 'l2']):
            break
        

    # Decode sequence to DFS code
    dfs_code = []
    for i in range(1, len(v1_seq) - 1):  # Skip the first token (start token) and last token (end token)
        try:
            v1 = int(node_id_vocab.get_token(v1_seq[i]))
            v2 = int(node_id_vocab.get_token(v2_seq[i]))
            l1 = node_label_vocab.get_token(l1_seq[i])
            e  = edge_label_vocab.get_token(e_seq[i])
            l2 = node_label_vocab.get_token(l2_seq[i])
            dfs_code.append((v1, v2, l1, e, l2))
        except:
            break

    # Convert to NetworkX graph
    graph = graph_from_dfscode(dfs_code)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if len(graph.nodes):
        max_comp = max(nx.connected_components(graph), key=len)
        graph = nx.Graph(graph.subgraph(max_comp))

    return graph