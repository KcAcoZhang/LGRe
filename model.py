# Acknowledgement: The basic framework is referenced from SANE (https://github.com/codeofpaper/SANe).
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

class LGRe(torch.nn.Module):
    def __init__(self, params):
        super(LGRe, self).__init__()
        self.p = params
        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim, padding_idx=None)
        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2, self.p.embed_dim, padding_idx=None)
        self.year_embed = torch.nn.Embedding(self.p.n_year, self.p.embed_dim, padding_idx=None)
        self.month_embed = torch.nn.Embedding(self.p.n_month, self.p.embed_dim, padding_idx=None)
        self.day_embed = torch.nn.Embedding(self.p.n_day, self.p.embed_dim, padding_idx=None)

        self.register_parameter('bias', nn.Parameter(torch.zeros(self.p.num_ent)))

        self.bceloss = torch.nn.BCELoss()
        self.chequer_perm = self.p.chequer_perm

        # ----
        self.inp_drop = nn.Dropout(self.p.inp_drop)
        self.feature_drop_l1 = nn.Dropout(self.p.feat_drop)
        self.feature_drop_l2 = nn.Dropout(self.p.feat_drop)
        self.feature_drop_l3 = nn.Dropout(self.p.feat_drop)
        self.hidden_drop = nn.Dropout(self.p.hid_drop)
        self.num_filt = [self.p.num_filt, self.p.num_filt, self.p.num_filt]
        self.ker_sz = [self.p.ker_sz, self.p.ker_sz, self.p.ker_sz]

        flat_sz_h = self.p.k_h
        flat_sz_w = 2 * self.p.k_w
        self.padding = 0
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt[-1]

        self.bnl0 = nn.BatchNorm2d(1)
        self.bnl2 = nn.BatchNorm2d(self.num_filt[1])
        self.bnl3 = nn.BatchNorm2d(self.num_filt[2])
        self.bnl1 = nn.BatchNorm2d(self.num_filt[0])
        self.bnfn = nn.BatchNorm1d(self.p.embed_dim)
        # ----
        self.param_gene_l1 = nn.Linear(self.p.embed_dim, self.num_filt[0] * 1 * self.ker_sz[0] * self.ker_sz[0])
        self.param_gene_l2 = nn.Linear(self.p.embed_dim,
                                       self.num_filt[1] * self.num_filt[0] * self.ker_sz[1] * self.ker_sz[1])
        self.param_gene_l3 = nn.Linear(self.p.embed_dim,
                                       self.num_filt[2] * self.num_filt[1] * self.ker_sz[2] * self.ker_sz[2])

        self.fc_y = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.fc_m = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

        self.time_encoder = torch.nn.GRU(input_size=self.p.embed_dim, hidden_size=self.p.embed_dim, bidirectional=False,
                                         batch_first=True)
        self.init_weights()

        #* add
        self.smiplelinear_y = nn.Linear(self.p.embed_dim, 1, bias=False)
        self.smiplelinear_m = nn.Linear(self.p.embed_dim, 1, bias=False)
        self.smiplelinear_d = nn.Linear(self.p.embed_dim, 1, bias=False)

        self.simple_t = nn.Linear(3 * self.p.embed_dim, self.p.embed_dim)
        self.simple_r = nn.Linear(2 * self.p.embed_dim, self.p.embed_dim)
        self.act_t = nn.LeakyReLU(0.2)
        self.act_r = nn.LeakyReLU(0.2)
    def init_weights(self):
        nn.init.xavier_normal_(self.ent_embed.weight)
        nn.init.xavier_normal_(self.rel_embed.weight)
        nn.init.xavier_normal_(self.year_embed.weight)
        nn.init.xavier_normal_(self.month_embed.weight)
        nn.init.xavier_normal_(self.day_embed.weight)

    def forward(self, sub, rel, year, month, day, neg_ents, strategy='one_to_x'):
        h_emb = self.ent_embed(sub)
        r_emb = self.rel_embed(rel)
        y_emb = self.year_embed(year)
        m_emb = self.month_embed(month)
        d_emb = self.day_embed(day)
        time_emb = self.time_encoder(torch.stack([y_emb, m_emb, d_emb], 1))[0]
        y_emb = time_emb[:, 0, :]
        m_emb = time_emb[:, 1, :]
        d_emb = time_emb[:, 2, :]
        #* ablation RU
        t_emb = self.act_t(self.simple_t(torch.cat([y_emb, m_emb, d_emb], dim=-1)))
        r_emb = self.act_r(self.simple_r(torch.cat([r_emb, t_emb], dim=-1)))
        comb_emb = torch.cat([h_emb, r_emb], dim=1)

        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        stack_inp = self.bnl0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = self.circular_padding_chw(x, self.ker_sz[0] // 2)
        #* year
        comb_y = self.smiplelinear_y(y_emb)
        y_emb = self.param_gene_l1(y_emb)
        y_emb = y_emb.view(-1, self.num_filt[0], 1, self.ker_sz[0], self.ker_sz[0])
        Batch, FN, C, FH, FW = y_emb.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * 0 - FH) // 1)
        out_w = int(1 + (W + 2 * 0 - FW) // 1)
        x = F.unfold(x, (self.ker_sz[0], self.ker_sz[0]))
        x = torch.bmm(x.transpose(1, 2), y_emb.view(Batch, y_emb.size(1), -1).transpose(1, 2)).transpose(1, 2)
        x = F.fold(x, (out_h, out_w), (1, 1))
        x = self.bnl1(x)
        x = torch.relu(x)

        x = self.feature_drop_l1(x)
        #* Store the emb of year
        x_y = self.fc_y(x.view(sub.size(0), -1))
        x = self.circular_padding_chw(x, self.ker_sz[1] // 2)

        #* month
        comb_m = self.smiplelinear_m(m_emb)
        m_emb = self.param_gene_l2(m_emb)
        m_emb = m_emb.view(-1, self.num_filt[1], self.num_filt[0], self.ker_sz[1], self.ker_sz[1])
        Batch, FN, C, FH, FW = m_emb.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * 0 - FH) // 1)
        out_w = int(1 + (W + 2 * 0 - FW) // 1)
        x = F.unfold(x, (self.ker_sz[1], self.ker_sz[1]))
        x = torch.bmm(x.transpose(1, 2), m_emb.view(Batch, m_emb.size(1), -1).transpose(1, 2)).transpose(1, 2)
        x = F.fold(x, (out_h, out_w), (1, 1))
        x = self.bnl2(x)
        x = torch.relu(x)

        x = self.feature_drop_l2(x)
        #* Store the emb of month
        x_m = self.fc_m(x.view(sub.size(0), -1))
        x = self.circular_padding_chw(x, self.ker_sz[2] // 2)

        #* day
        comb_d = self.smiplelinear_d(d_emb)
        d_emb = self.param_gene_l3(d_emb)
        d_emb = d_emb.view(-1, self.num_filt[2], self.num_filt[1], self.ker_sz[2], self.ker_sz[2])
        Batch, FN, C, FH, FW = d_emb.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * 0 - FH) // 1)
        out_w = int(1 + (W + 2 * 0 - FW) // 1)
        x = F.unfold(x, (self.ker_sz[2], self.ker_sz[2]))
        x = torch.bmm(x.transpose(1, 2), d_emb.view(Batch, d_emb.size(1), -1).transpose(1, 2)).transpose(1, 2)
        x = F.fold(x, (out_h, out_w), (1, 1))
        x = self.bnl3(x)
        x = torch.relu(x)
        x = self.feature_drop_l3(x)

        #* the emb of day
        x = x.view(sub.size(0), -1)

        #* set adaptive weight for emb of year, month, and day
        comb_time_emb = torch.cat([comb_y, comb_m, comb_d], dim=1)
        total_w = torch.softmax(comb_time_emb, dim=1)
        x = self.fc(x)
        wy = total_w[:,0].unsqueeze(1).repeat(1, x.shape[1])
        wm = total_w[:,1].unsqueeze(1).repeat(1, x.shape[1])
        wd = total_w[:,2].unsqueeze(1).repeat(1, x.shape[1])
        x = wy * x_y + wm * x_m + wd * x
        x = self.hidden_drop(x)
        x = self.bnfn(x)
        x = torch.relu(x)

        if strategy == 'one_to_n':
            x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
            x += self.bias[neg_ents]
        pred = torch.sigmoid(x)
        loss_t = self.temporal_emb(self.year_embed, self.month_embed, self.day_embed)
        return pred, loss_t

    def loss(self, pred, true_label):
        loss = self.bceloss(pred, true_label)
        return loss

    def temporal_emb(self, y_emb, m_emb, d_emb):
        if self.p.dataset == 'icews14':
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            combined_list = []
            init_day_data = d_emb.weight
            init_month_data = m_emb.weight
            init_y_data = y_emb.weight
            for month_index in range(self.p.n_month):
                days_count = torch.tensor([days_in_month[month_index]]).cuda()
                y_data = init_y_data.repeat(days_count, 1)
                month_data = init_month_data[month_index].repeat(days_count, 1)
                day_data = init_day_data[:days_count]
                combined_data = self.act_t(self.simple_t(torch.cat((y_data, month_data, day_data), dim=1)))
                # Append the relevant slice from the days_tensor
                combined_list.append(combined_data)

                # Concatenate the slices to form the final [365, 200] tensor
            final_tensor = torch.vstack(combined_list)
        elif self.p.dataset == 'wikidata' or self.p.dataset == 'yago':
            y_data = y_emb.weight
            month_data = m_emb.weight.repeat(y_emb.weight.shape[0], 1)
            day_data = d_emb.weight.repeat(y_emb.weight.shape[0], 1)
            final_tensor = self.act_t(self.simple_t(torch.cat((y_data, month_data, day_data), dim=1)))
        elif self.p.dataset == 'icews05-15':
            # Base array indicating the number of days in each month for a non-leap year
            days_in_month_non_leap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            # Adjust for leap years
            days_in_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

            # Leap years in the range 2005-2015
            leap_years = [2008, 2012]

            # Initialize an empty list to store the combined slices
            combined_list = []
            init_year_data = y_emb.weight
            init_month_data = m_emb.weight
            init_day_data = d_emb.weight
            # Iterate through each year
            for year_index in range(self.p.n_year):
                year = 2005 + year_index
                year_data = init_year_data[year_index].unsqueeze(0)
                # Determine if the current year is a leap year
                if year in leap_years:
                    days_in_month = days_in_month_leap
                else:
                    days_in_month = days_in_month_non_leap
                # Iterate through each month
                for month_index in range(12):
                    days_count = days_in_month[month_index]
                    # Extract the month tensor (repeating it for each day in the month)
                    month_data = init_month_data[month_index].unsqueeze(0).repeat(days_count, 1)
                    # Extract the relevant slice from the days_tensor
                    day_data = init_day_data[:days_count]
                    # Repeat the year data for the number of days in the month
                    repeated_year_data = year_data.repeat(days_count, 1)
                    # Combine year, month, and day data
                    combined_data = self.act_t(self.simple_t(torch.cat((repeated_year_data, month_data, day_data), dim=1)))
                    combined_list.append(combined_data)
            # Concatenate the combined data to form the final tensor
            final_tensor = torch.vstack(combined_list)
        else:
            raise ('no this dataset')
        return final_tensor

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded