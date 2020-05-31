import torch
import torch.nn as nn
import torch.nn.functional as F

class PLA_Attention_Model(nn.Module):
    def __init__(self, byte_hidden_size, packet_hidden_size, packet_output_size, max_packet=1500, max_flow=128):
        super().__init__()
        self.byte_emb_size = 50
        self.byte_hidden_size = byte_hidden_size
        self.packet_emb_size = 50
        self.packet_hidden_size = packet_hidden_size
        self.packet_output_size = packet_output_size  
        self.max_packet = max_packet
        self.max_flow = max_flow

        self.embedding = nn.Embedding(256 * 256, self.byte_emb_size)
        self.byte_GRU = nn.GRU(self.byte_emb_size, self.byte_hidden_size, bias=False, bidirectional=True)
        self.byte_attn = nn.Linear(self.byte_hidden_size * 2, self.packet_emb_size)
        self.packet_GRU = nn.GRU(self.packet_emb_size, self.packet_hidden_size, bias=False, bidirectional=True)
        self.packet_attn = nn.Linear(self.packet_hidden_size * 2, self.packet_output_size)
        self.classify = nn.Linear(self.packet_output_size, 11)

    # lstm_output : [batch_size, n_step, self.byte_hidden_size * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.byte_hidden_size * 2, 1)   # hidden : [batch_size, self.byte_hidden_size * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, self.byte_hidden_size * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, self.byte_hidden_size * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights # context : [batch_size, self.byte_hidden_size * num_directions(=2)]

    def forward(self, flow):
        num_packet = flow.shape[0]
        batch_size = flow.shape[1]
        embedded_bytes_list = self.embedding(flow)
        # print(embedded_bytes_list.shape)
        encoding_bytes_list = torch.zeros((num_packet, batch_size, self.packet_emb_size)).cuda()
        for i, embedded_bytes in enumerate(embedded_bytes_list):
            h0_bytes = torch.randn(2, batch_size, self.byte_hidden_size).cuda()
            embedded_bytes = embedded_bytes.transpose(0,1)
            # print(embedded_bytes.shape, h0_bytes.shape)
            output, final_hidden_state = self.byte_GRU(embedded_bytes, h0_bytes)
            output = output.permute(1, 0, 2)
            # print(output.shape)
            # print(final_hidden_state.shape)
            attn_output, attention = self.attention_net(output, final_hidden_state)
            # print(attn_output.shape)
            encoding_bytes_list[i] = self.byte_attn(attn_output)

        # print(encoding_bytes_list.shape)
        h0_packet = torch.randn(2, batch_size, self.packet_hidden_size).cuda()
        # print(h0_packet.shape)
        output, final_hidden_state = self.packet_GRU(encoding_bytes_list, h0_packet)
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # print(attn_output.shape)
        output = self.packet_attn(attn_output)
        classify = self.classify(output)

        return classify
        

if __name__ == '__main__':
    batch_size = 3
    model = PLA_Attention_Model(100,100,50).cuda()
    data = torch.randint(255, size=(4, batch_size, 120)).cuda() # flow_len, batch_size, packet_len
    print(data.shape)
    res = model(data)
    print(res.shape)
